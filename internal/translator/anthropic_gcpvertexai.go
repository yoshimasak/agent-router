// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"google.golang.org/genai"
	"k8s.io/utils/ptr"

	"github.com/envoyproxy/ai-gateway/internal/apischema/anthropic"
	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

const (
	mimeTypeApplicationPDF = "application/pdf"

	// geminiThoughtSignaturePrefix marks thinking block signatures minted by this translator.
	// Anthropic-issued signatures are valid base64 as well, so the prefix is the only reliable way
	// to avoid forwarding them to Gemini.
	geminiThoughtSignaturePrefix = "gemini:"

	// Upper bounds (exclusive) of Anthropic thinking budgets mapped to Gemini 3 thinking levels.
	// Claude Code uses 4000 ("think"), 10000 ("think hard") and 31999 ("ultrathink").
	anthropicThinkingBudgetLowMax    = 8192
	anthropicThinkingBudgetMediumMax = 24576

	anthropicThinkingDisplayOmitted = "omitted"
)

var anthropicToolUseIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)

// NewAnthropicToGCPVertexAITranslator implements [Factory] for Anthropic Messages to GCP Vertex AI Gemini translation.
func NewAnthropicToGCPVertexAITranslator(modelNameOverride internalapi.ModelNameOverride) AnthropicMessagesTranslator {
	return &anthropicToGCPVertexAITranslator{modelNameOverride: modelNameOverride}
}

// anthropicToGCPVertexAITranslator translates the Anthropic Messages API to the native Gemini
// generateContent / streamGenerateContent API on GCP Vertex AI.
//
// Gemini thought signatures are carried in Anthropic thinking blocks: a thinking block is placed
// right before the tool_use block whose function call owned the signature. Signatures on thought,
// text or empty parts are held and given to the next function call without its own signature, and
// dropped when no function call follows. Such signatures are only recommended to be sent back, while
// a trailing thinking block would hide the final text from clients reading the last block. Thinking
// blocks without a Gemini signature (thought summaries) carry the bare "gemini:" prefix, so that every
// thinking block produced by this translator can be identified by the prefix. On the request side, the
// signature of a thinking block is restored onto the next text or tool_use part, and a trailing one onto
// the last part; a "gemini:" signature with an empty payload is treated as no signature.
// See https://cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures.
//
// Request fields without a Gemini counterpart are ignored: metadata, service_tier, cache_control,
// tool_choice.disable_parallel_tool_use, container, mcp_servers, context_management and safeguards.
// Server and client built-in tools are skipped (see anthropicToolsToGemini), max_tokens is passed through
// unclamped, the MIME type of an image URL is guessed from its extension (JPEG by default) and a document
// URL is assumed to be a PDF. The count_tokens endpoint is not supported.
type anthropicToGCPVertexAITranslator struct {
	modelNameOverride internalapi.ModelNameOverride
	requestModel      internalapi.RequestModel
	stream            bool
	streamState       *geminiToAnthropicStreamState
	debugLogEnabled   bool
	enableRedaction   bool
	logger            *slog.Logger
}

// RequestBody implements [AnthropicMessagesTranslator.RequestBody].
func (a *anthropicToGCPVertexAITranslator) RequestBody(raw []byte, body *anthropic.MessagesRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	a.stream = body.Stream
	a.requestModel = cmp.Or(a.modelNameOverride, body.Model)

	// output_config is not part of anthropic.MessagesRequest, so it is read from the raw body.
	outputConfig, err := parseAnthropicOutputConfig(raw)
	if err != nil {
		return nil, nil, err
	}
	gcpReq, err := anthropicToGeminiRequest(body, a.requestModel, outputConfig, a.logger)
	if err != nil {
		return nil, nil, err
	}
	newBody, err = json.Marshal(gcpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal Gemini request: %w", err)
	}

	var p string
	if a.stream {
		p = buildGCPModelPathSuffix(gcpModelPublisherGoogle, a.requestModel, gcpMethodStreamGenerateContent, "alt=sse")
		a.streamState = &geminiToAnthropicStreamState{requestModel: a.requestModel}
	} else {
		p = buildGCPModelPathSuffix(gcpModelPublisherGoogle, a.requestModel, gcpMethodGenerateContent)
	}
	newHeaders = []internalapi.Header{
		{pathHeaderName, p},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// ResponseHeaders implements [AnthropicMessagesTranslator.ResponseHeaders].
func (a *anthropicToGCPVertexAITranslator) ResponseHeaders(_ map[string]string) (newHeaders []internalapi.Header, err error) {
	if a.stream {
		newHeaders = []internalapi.Header{{contentTypeHeaderName, eventStreamContentType}}
	}
	return
}

// ResponseBody implements [AnthropicMessagesTranslator.ResponseBody].
func (a *anthropicToGCPVertexAITranslator) ResponseBody(_ map[string]string, body io.Reader, endOfStream bool, span tracingapi.MessageSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel string, err error,
) {
	if a.stream {
		return a.responseBodyStreaming(body, endOfStream, span)
	}
	return a.responseBodyNonStreaming(body, span)
}

func (a *anthropicToGCPVertexAITranslator) responseBodyNonStreaming(body io.Reader, span tracingapi.MessageSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel string, err error,
) {
	responseModel = a.requestModel
	gcpResp := &genai.GenerateContentResponse{}
	if err = json.NewDecoder(body).Decode(gcpResp); err != nil {
		return nil, nil, tokenUsage, responseModel, fmt.Errorf("failed to decode Gemini response: %w", err)
	}
	responseModel = cmp.Or(gcpResp.ModelVersion, a.requestModel)

	anthropicResp, tokenUsage, failure := geminiResponseToAnthropic(gcpResp, responseModel)
	if failure != "" {
		// The upstream status is 200, so the status is rewritten along with the body. The tokens were
		// still consumed and are reported for metrics.
		newBody, err = json.Marshal(anthropicAPIErrorBody(failure))
		if err != nil {
			return nil, nil, tokenUsage, responseModel, fmt.Errorf("failed to marshal error body: %w", err)
		}
		newHeaders = []internalapi.Header{
			{statusHeaderName, strconv.Itoa(http.StatusInternalServerError)},
			{contentTypeHeaderName, jsonContentType},
			{contentLengthHeaderName, strconv.Itoa(len(newBody))},
		}
		return
	}

	if a.debugLogEnabled && a.enableRedaction && a.logger != nil {
		redactedResp := a.RedactAnthropicBody(anthropicResp)
		if jsonBody, marshalErr := json.Marshal(redactedResp); marshalErr == nil {
			a.logger.Debug("response body processing", slog.Any("response", string(jsonBody)))
		}
	}
	if span != nil {
		span.RecordResponse(anthropicResp)
	}

	newBody, err = json.Marshal(anthropicResp)
	if err != nil {
		return nil, nil, tokenUsage, responseModel, fmt.Errorf("failed to marshal Anthropic response: %w", err)
	}
	newHeaders = []internalapi.Header{{contentLengthHeaderName, strconv.Itoa(len(newBody))}}
	return
}

func (a *anthropicToGCPVertexAITranslator) responseBodyStreaming(body io.Reader, endOfStream bool, span tracingapi.MessageSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel string, err error,
) {
	responseModel = a.requestModel
	s := a.streamState
	if s == nil {
		return nil, nil, tokenUsage, responseModel, errors.New("stream state not initialized")
	}
	data, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, tokenUsage, responseModel, fmt.Errorf("failed to read stream body: %w", err)
	}

	s.span = span
	// A non-nil empty body replaces the upstream chunk with nothing instead of passing it through.
	out := make([]byte, 0)
	if err = s.process(data, endOfStream, &out); err != nil {
		return nil, nil, tokenUsage, responseModel, err
	}
	return nil, out, s.tokenUsage, cmp.Or(s.model, a.requestModel), nil
}

// ResponseError implements [AnthropicMessagesTranslator.ResponseError].
func (a *anthropicToGCPVertexAITranslator) ResponseError(respHeaders map[string]string, body io.Reader) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	buf, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read error body: %w", err)
	}
	statusCode, _ := strconv.Atoi(respHeaders[statusHeaderName])
	message := string(buf)
	if details, ok := parseGCPVertexAIError(buf); ok {
		message = gcpVertexAIErrorMessage(details)
		if statusCode == 0 {
			statusCode = details.Code
		}
	}

	newBody, err = json.Marshal(anthropic.ErrorResponse{
		Type:  "error",
		Error: anthropic.ErrorResponseMessage{Type: anthropicErrorTypeForHTTPStatus(statusCode), Message: message},
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal error body: %w", err)
	}
	newHeaders = []internalapi.Header{
		{contentTypeHeaderName, jsonContentType},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// SetRedactionConfig implements [AnthropicResponseRedactor.SetRedactionConfig].
func (a *anthropicToGCPVertexAITranslator) SetRedactionConfig(debugLogEnabled, enableRedaction bool, logger *slog.Logger) {
	a.debugLogEnabled = debugLogEnabled
	a.enableRedaction = enableRedaction
	a.logger = logger
}

// RedactAnthropicBody implements [AnthropicResponseRedactor.RedactAnthropicBody].
func (a *anthropicToGCPVertexAITranslator) RedactAnthropicBody(resp *anthropic.MessagesResponse) *anthropic.MessagesResponse {
	if resp == nil {
		return nil
	}
	redacted := *resp
	if len(resp.Content) > 0 {
		redacted.Content = make([]anthropic.MessagesContentBlock, len(resp.Content))
		for i := range resp.Content {
			redacted.Content[i] = redactAnthropicContent(&resp.Content[i])
		}
	}
	return &redacted
}

// -------------------------------------------------------------
// Request conversion: Anthropic Messages -> Gemini
// -------------------------------------------------------------

// anthropicOutputConfig is the output_config of a Messages request, which anthropic.MessagesRequest does
// not carry. jsonSchema is the schema of a json_schema format and nil when no format is requested.
type anthropicOutputConfig struct {
	effort     string
	jsonSchema map[string]any
}

// parseAnthropicOutputConfig reads output_config from the raw request body. json_schema is the only format
// defined by the Anthropic API; any other format type is rejected rather than silently ignored, since the
// client would otherwise receive plain text where it expects JSON.
func parseAnthropicOutputConfig(raw []byte) (anthropicOutputConfig, error) {
	oc := anthropicOutputConfig{effort: gjson.GetBytes(raw, "output_config.effort").Str}
	format := gjson.GetBytes(raw, "output_config.format")
	if !format.Exists() || format.Type == gjson.Null {
		return oc, nil
	}
	if !format.IsObject() {
		return oc, fmt.Errorf("%w: output_config.format must be an object", internalapi.ErrInvalidRequestBody)
	}
	if typ := format.Get("type").Str; typ != "json_schema" {
		return oc, fmt.Errorf("%w: unsupported output_config.format.type %q (supported: json_schema)", internalapi.ErrInvalidRequestBody, typ)
	}
	schema := format.Get("schema")
	if !schema.IsObject() {
		return oc, fmt.Errorf("%w: output_config.format.schema must be a JSON object", internalapi.ErrInvalidRequestBody)
	}
	if err := json.Unmarshal([]byte(schema.Raw), &oc.jsonSchema); err != nil {
		return oc, fmt.Errorf("%w: invalid output_config.format.schema: %w", internalapi.ErrInvalidRequestBody, err)
	}
	return oc, nil
}

func anthropicToGeminiRequest(
	body *anthropic.MessagesRequest, model internalapi.RequestModel, outputConfig anthropicOutputConfig, logger *slog.Logger,
) (*gcp.GenerateContentRequest, error) {
	// Gemini has no counterpart of assistant prefill: a request ending with a model content is rejected.
	if n := len(body.Messages); n > 0 && body.Messages[n-1].Role == anthropic.MessageRoleAssistant {
		return nil, errAnthropicPrefill
	}
	contents, err := anthropicMessagesToGeminiContents(body.Messages)
	if err != nil {
		return nil, err
	}
	// Empty messages are dropped during conversion, so the checks are repeated on the result: Vertex AI
	// rejects both an empty contents list and one ending with a model content.
	if len(contents) == 0 {
		return nil, fmt.Errorf("%w: messages must contain at least one non-empty message", internalapi.ErrInvalidRequestBody)
	}
	if contents[len(contents)-1].Role == genai.RoleModel {
		return nil, errAnthropicPrefill
	}
	tools, skippedTools, err := anthropicToolsToGemini(body.Tools, responseJSONSchemaAvailable(model))
	if err != nil {
		return nil, err
	}
	if len(skippedTools) > 0 {
		// Dropping every tool would let the model answer without the tools the client relies on.
		if len(tools) == 0 {
			return nil, fmt.Errorf("%w: server tools such as web_search are not supported for GCPVertexAI backends (got %s)",
				internalapi.ErrInvalidRequestBody, strings.Join(skippedTools, ", "))
		}
		if logger != nil {
			logger.Debug("skipping Anthropic built-in tools not supported by GCPVertexAI backends",
				slog.Any("tool_types", skippedTools))
		}
	}
	if tc := body.ToolChoice; tc != nil && tc.Tool != nil && !declaresFunction(tools, tc.Tool.Name) {
		return nil, fmt.Errorf("%w: tool_choice refers to tool %q, which is not a custom tool in tools",
			internalapi.ErrInvalidRequestBody, tc.Tool.Name)
	}
	generationConfig, err := anthropicToGeminiGenerationConfig(body, model, outputConfig)
	if err != nil {
		return nil, err
	}
	req := &gcp.GenerateContentRequest{
		Contents:          contents,
		Tools:             tools,
		SystemInstruction: anthropicSystemToGemini(body.System),
		GenerationConfig:  generationConfig,
	}
	if len(tools) > 0 {
		req.ToolConfig = anthropicToolChoiceToGemini(body.ToolChoice)
	}
	return req, nil
}

var errAnthropicPrefill = fmt.Errorf("%w: assistant prefill is not supported for GCPVertexAI backends", internalapi.ErrInvalidRequestBody)

// declaresFunction reports whether a function with the given name is declared in tools.
func declaresFunction(tools []genai.Tool, name string) bool {
	for i := range tools {
		for _, decl := range tools[i].FunctionDeclarations {
			if decl.Name == name {
				return true
			}
		}
	}
	return false
}

// anthropicMessageRoleSystem is the role of mid-conversation system messages sent by Claude Code.
const anthropicMessageRoleSystem anthropic.MessageRole = "system"

func anthropicMessagesToGeminiContents(messages []anthropic.MessageParam) ([]genai.Content, error) {
	// toolNames maps tool_use ids to tool names as the assistant messages are converted, so that a
	// tool_result can only refer to a tool_use that precedes it.
	toolNames := make(map[string]string)
	var contents []genai.Content
	// nonPromptParts tracks user parts that do not start a new turn: text converted from system messages
	// and parts split off from contents carrying function responses.
	nonPromptParts := make(map[*genai.Part]struct{})
	for i := range messages {
		msg := &messages[i]
		var role string
		var parts []*genai.Part
		var err error
		switch msg.Role {
		case anthropic.MessageRoleUser:
			role = genai.RoleUser
			parts, err = anthropicUserContentToGeminiParts(&msg.Content, toolNames)
		case anthropic.MessageRoleAssistant:
			role = genai.RoleModel
			parts = anthropicAssistantContentToGeminiParts(&msg.Content)
			for _, part := range parts {
				if fc := part.FunctionCall; fc != nil && fc.Name != "" {
					toolNames[fc.ID] = fc.Name
				}
			}
		case anthropicMessageRoleSystem:
			// Gemini has no mid-conversation system role, so the text is passed as user text in place.
			role = genai.RoleUser
			if text := anthropicSystemMessageText(&msg.Content); text != "" {
				part := genai.NewPartFromText(text)
				nonPromptParts[part] = struct{}{}
				parts = []*genai.Part{part}
			}
		default:
			return nil, fmt.Errorf("%w: unsupported role %q in message %d", internalapi.ErrInvalidRequestBody, msg.Role, i)
		}
		if err != nil {
			return nil, fmt.Errorf("invalid message %d: %w", i, err)
		}
		if len(parts) == 0 {
			continue
		}
		// Gemini rejects consecutive contents of the same role in some cases, and the Anthropic API
		// merges them anyway.
		if n := len(contents); n > 0 && contents[n-1].Role == role {
			contents[n-1].Parts = append(contents[n-1].Parts, parts...)
			continue
		}
		contents = append(contents, genai.Content{Role: role, Parts: parts})
	}
	contents = splitFunctionResponseContents(contents, nonPromptParts)
	fillMissingCurrentTurnThoughtSignatures(contents, nonPromptParts)
	return contents, nil
}

// anthropicSystemMessageText concatenates the text blocks of a system message; other blocks are ignored.
func anthropicSystemMessageText(content *anthropic.MessageContent) string {
	if content.Text != "" {
		return content.Text
	}
	var texts []string
	for i := range content.Array {
		if tb := content.Array[i].Text; tb != nil && tb.Text != "" {
			texts = append(texts, tb.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// splitFunctionResponseContents splits every user content mixing function responses with other parts into
// a content with only the function responses followed by a content with the rest, since Vertex AI rejects
// a user content with text after function responses ("Requests ending with a model turn are not
// supported."). The split-off parts are recorded in nonPromptParts so that they do not start a new turn.
func splitFunctionResponseContents(contents []genai.Content, nonPromptParts map[*genai.Part]struct{}) []genai.Content {
	out := make([]genai.Content, 0, len(contents))
	for _, c := range contents {
		if c.Role != genai.RoleUser {
			out = append(out, c)
			continue
		}
		var frs, rest []*genai.Part
		for _, part := range c.Parts {
			if part.FunctionResponse != nil {
				frs = append(frs, part)
			} else {
				rest = append(rest, part)
			}
		}
		if len(frs) == 0 || len(rest) == 0 {
			out = append(out, c)
			continue
		}
		for _, part := range rest {
			nonPromptParts[part] = struct{}{}
		}
		out = append(out, genai.Content{Role: genai.RoleUser, Parts: frs}, genai.Content{Role: genai.RoleUser, Parts: rest})
	}
	return out
}

func anthropicUserContentToGeminiParts(content *anthropic.MessageContent, toolNames map[string]string) ([]*genai.Part, error) {
	if content.Text != "" {
		return []*genai.Part{genai.NewPartFromText(content.Text)}, nil
	}
	var parts []*genai.Part
	for i := range content.Array {
		block := &content.Array[i]
		if block.ToolResult != nil {
			part, err := anthropicToolResultToGeminiPart(block.ToolResult, toolNames)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
			continue
		}
		pieces, err := anthropicBlockToGeminiPieces(block.Text, block.Image, block.Document, block.SearchResult)
		if err != nil {
			return nil, err
		}
		for _, piece := range pieces {
			if piece.media != nil {
				parts = append(parts, piece.media.part())
			} else if piece.text != "" {
				parts = append(parts, genai.NewPartFromText(piece.text))
			}
		}
	}
	return parts, nil
}

// anthropicAssistantContentToGeminiParts converts an assistant message and restores the Gemini thought
// signatures carried by its thinking blocks. Thinking text is not sent back since the signature is what
// preserves the reasoning context, except when the message holds nothing else: the thinking text is then
// sent as a thought part so that the model turn is kept. Vertex AI rejects a thought part without text, so
// a message whose thinking blocks are all empty (thinking.display: omitted) is dropped together with its
// signature. redacted_thinking blocks and non-Gemini signatures are dropped.
func anthropicAssistantContentToGeminiParts(content *anthropic.MessageContent) []*genai.Part {
	if content.Text != "" {
		return []*genai.Part{genai.NewPartFromText(content.Text)}
	}
	var parts []*genai.Part
	var pendingSig []byte
	var thinkingTexts []string
	for i := range content.Array {
		block := &content.Array[i]
		var part *genai.Part
		switch {
		case block.Thinking != nil:
			if sig := decodeGeminiThoughtSignature(block.Thinking.Signature); sig != nil {
				pendingSig = sig
			}
			if block.Thinking.Thinking != "" {
				thinkingTexts = append(thinkingTexts, block.Thinking.Thinking)
			}
			continue
		case block.Text != nil:
			if block.Text.Text == "" {
				continue
			}
			part = genai.NewPartFromText(block.Text.Text)
		case block.ToolUse != nil:
			part = &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   block.ToolUse.ID,
				Name: block.ToolUse.Name,
				Args: block.ToolUse.Input,
			}}
		default:
			continue
		}
		part.ThoughtSignature, pendingSig = pendingSig, nil
		parts = append(parts, part)
	}
	if len(parts) == 0 {
		if len(thinkingTexts) == 0 {
			return nil
		}
		return []*genai.Part{{Text: strings.Join(thinkingTexts, "\n"), Thought: true, ThoughtSignature: pendingSig}}
	}
	// Gemini parts must carry data, so a trailing signature goes onto the last part instead of an empty one.
	if last := len(parts) - 1; pendingSig != nil && len(parts[last].ThoughtSignature) == 0 {
		parts[last].ThoughtSignature = pendingSig
	}
	return parts
}

// fillMissingCurrentTurnThoughtSignatures adds the documented dummy signature to the first functionCall
// of every model step in the current turn when the client did not echo a Gemini signature back, since
// Gemini 3 rejects such requests. The current turn begins after the most recent user content that is a
// real user prompt: function responses, text split off from them (e.g. Claude Code reminders) and
// system-derived text are treated as part of the turn so that no step is left without a signature.
func fillMissingCurrentTurnThoughtSignatures(contents []genai.Content, nonPromptParts map[*genai.Part]struct{}) {
	start := 0
	for i := len(contents) - 1; i >= 0; i-- {
		if contents[i].Role == genai.RoleUser && isUserPromptContent(contents[i].Parts, nonPromptParts) {
			start = i + 1
			break
		}
	}
	for i := start; i < len(contents); i++ {
		if contents[i].Role != genai.RoleModel {
			continue
		}
		for _, part := range contents[i].Parts {
			if part.FunctionCall != nil {
				if len(part.ThoughtSignature) == 0 {
					part.ThoughtSignature = dummyThoughtSignature
				}
				break
			}
		}
	}
}

// isUserPromptContent reports whether a user content starts a new turn: it has no function response and
// at least one part not recorded in nonPromptParts.
func isUserPromptContent(parts []*genai.Part, nonPromptParts map[*genai.Part]struct{}) bool {
	prompt := false
	for _, part := range parts {
		if part.FunctionResponse != nil {
			return false
		}
		if _, ok := nonPromptParts[part]; !ok {
			prompt = true
		}
	}
	return prompt
}

// anthropicToolResultToGeminiPart converts a tool_result block to a functionResponse part. Images and
// documents are passed as multimodal function response parts, supported by Gemini 3 and later:
// https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#mm-fr
func anthropicToolResultToGeminiPart(tr *anthropic.ToolResultBlockParam, toolNames map[string]string) (*genai.Part, error) {
	name, ok := toolNames[tr.ToolUseID]
	if !ok {
		return nil, fmt.Errorf("%w: tool_result references unknown tool_use_id %q", internalapi.ErrInvalidRequestBody, tr.ToolUseID)
	}
	var texts []string
	var mediaParts []*genai.FunctionResponsePart
	if tr.Content != nil {
		if tr.Content.Text != "" {
			texts = append(texts, tr.Content.Text)
		}
		for i := range tr.Content.Array {
			item := &tr.Content.Array[i]
			pieces, err := anthropicBlockToGeminiPieces(item.Text, item.Image, item.Document, item.SearchResult)
			if err != nil {
				return nil, err
			}
			for _, piece := range pieces {
				if piece.media != nil {
					mediaParts = append(mediaParts, piece.media.functionResponsePart())
				} else if piece.text != "" {
					texts = append(texts, piece.text)
				}
			}
		}
	}
	key := "output"
	if tr.IsError {
		key = "error"
	}
	return &genai.Part{FunctionResponse: &genai.FunctionResponse{
		ID:       tr.ToolUseID,
		Name:     name,
		Response: map[string]any{key: strings.Join(texts, "\n")},
		Parts:    mediaParts,
	}}, nil
}

// geminiMedia is binary or URI-referenced content extracted from an Anthropic image or document block.
type geminiMedia struct {
	mimeType string
	data     []byte
	uri      string
}

func (m *geminiMedia) part() *genai.Part {
	if m.uri != "" {
		return genai.NewPartFromURI(m.uri, m.mimeType)
	}
	return genai.NewPartFromBytes(m.data, m.mimeType)
}

func (m *geminiMedia) functionResponsePart() *genai.FunctionResponsePart {
	if m.uri != "" {
		return genai.NewFunctionResponsePartFromURI(m.uri, m.mimeType)
	}
	return genai.NewFunctionResponsePartFromBytes(m.data, m.mimeType)
}

// geminiPiece holds either text or media.
type geminiPiece struct {
	text  string
	media *geminiMedia
}

// anthropicBlockToGeminiPieces converts the content block kinds shared by message content and tool
// results. At most one of the arguments is non-nil; unknown block kinds yield no pieces.
func anthropicBlockToGeminiPieces(
	text *anthropic.TextBlockParam,
	image *anthropic.ImageBlockParam,
	document *anthropic.DocumentBlockParam,
	searchResult *anthropic.SearchResultBlockParam,
) ([]geminiPiece, error) {
	switch {
	case text != nil:
		return []geminiPiece{{text: text.Text}}, nil
	case image != nil:
		media, err := anthropicImageSourceToGeminiMedia(&image.Source)
		if err != nil {
			return nil, err
		}
		return []geminiPiece{{media: media}}, nil
	case document != nil:
		return anthropicDocumentSourceToGeminiPieces(&document.Source)
	case searchResult != nil:
		return []geminiPiece{{text: anthropicSearchResultToText(searchResult)}}, nil
	}
	return nil, nil
}

func anthropicImageSourceToGeminiMedia(src *anthropic.ImageSource) (*geminiMedia, error) {
	switch {
	case src.Base64 != nil:
		// An empty URL or data would otherwise become a 0-byte inlineData part, which Vertex AI rejects.
		if src.Base64.MediaType == "" || src.Base64.Data == "" {
			return nil, fmt.Errorf("%w: base64 image source requires media_type and data", internalapi.ErrInvalidRequestBody)
		}
		data, err := base64.StdEncoding.DecodeString(src.Base64.Data)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid base64 image data", internalapi.ErrInvalidRequestBody)
		}
		return &geminiMedia{mimeType: src.Base64.MediaType, data: data}, nil
	case src.URL != nil:
		if src.URL.URL == "" {
			return nil, fmt.Errorf("%w: image url must not be empty", internalapi.ErrInvalidRequestBody)
		}
		return &geminiMedia{mimeType: imageMIMETypeFromURL(src.URL.URL), uri: src.URL.URL}, nil
	}
	// The source type is not kept by the unmarshaler for unknown sources such as "file" (Files API).
	return nil, fmt.Errorf("%w: unsupported image source type for GCPVertexAI backends (supported: base64, url)",
		internalapi.ErrInvalidRequestBody)
}

// imageMIMETypeFromURL guesses the MIME type from the extension of the URL path, ignoring the query
// and fragment, and falls back to JPEG.
func imageMIMETypeFromURL(rawURL string) string {
	if u, err := url.Parse(rawURL); err == nil {
		if mt := mime.TypeByExtension(path.Ext(u.Path)); mt != "" {
			return mt
		}
	}
	return mimeTypeImageJPEG
}

func anthropicDocumentSourceToGeminiPieces(src *anthropic.DocumentSource) ([]geminiPiece, error) {
	switch {
	case src.Base64PDF != nil:
		if src.Base64PDF.Data == "" {
			return nil, fmt.Errorf("%w: base64 document source requires data", internalapi.ErrInvalidRequestBody)
		}
		data, err := base64.StdEncoding.DecodeString(src.Base64PDF.Data)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid base64 document data", internalapi.ErrInvalidRequestBody)
		}
		return []geminiPiece{{media: &geminiMedia{mimeType: cmp.Or(src.Base64PDF.MediaType, mimeTypeApplicationPDF), data: data}}}, nil
	case src.PlainText != nil:
		return []geminiPiece{{text: src.PlainText.Data}}, nil
	case src.URL != nil:
		if src.URL.URL == "" {
			return nil, fmt.Errorf("%w: document url must not be empty", internalapi.ErrInvalidRequestBody)
		}
		return []geminiPiece{{media: &geminiMedia{mimeType: mimeTypeApplicationPDF, uri: src.URL.URL}}}, nil
	case src.ContentBlock != nil:
		c := &src.ContentBlock.Content
		if c.Text != "" {
			return []geminiPiece{{text: c.Text}}, nil
		}
		var pieces []geminiPiece
		for i := range c.Array {
			p, err := anthropicBlockToGeminiPieces(c.Array[i].Text, c.Array[i].Image, nil, nil)
			if err != nil {
				return nil, err
			}
			pieces = append(pieces, p...)
		}
		return pieces, nil
	}
	return nil, fmt.Errorf("%w: unsupported document source type for GCPVertexAI backends (supported: base64, text, url, content)",
		internalapi.ErrInvalidRequestBody)
}

func anthropicSearchResultToText(sr *anthropic.SearchResultBlockParam) string {
	lines := make([]string, 0, len(sr.Content)+2)
	if sr.Title != "" {
		lines = append(lines, sr.Title)
	}
	if sr.Source != "" {
		lines = append(lines, sr.Source)
	}
	for _, t := range sr.Content {
		lines = append(lines, t.Text)
	}
	return strings.Join(lines, "\n")
}

func anthropicSystemToGemini(system *anthropic.SystemPrompt) *genai.Content {
	if system == nil {
		return nil
	}
	var parts []*genai.Part
	if system.Text != "" {
		parts = append(parts, genai.NewPartFromText(system.Text))
	}
	for _, t := range system.Texts {
		if t.Text != "" {
			parts = append(parts, genai.NewPartFromText(t.Text))
		}
	}
	if len(parts) == 0 {
		return nil
	}
	return &genai.Content{Parts: parts}
}

// anthropicToolsToGemini converts custom tools to function declarations. Anthropic server and client
// built-in tools (bash, text editor, web search) have no Gemini equivalent and are skipped; their types
// are returned in skipped.
func anthropicToolsToGemini(tools []anthropic.ToolUnion, parametersJSONSchemaAvailable bool) (
	geminiTools []genai.Tool, skipped []string, err error,
) {
	var decls []*genai.FunctionDeclaration
	for i := range tools {
		t := tools[i].Tool
		if t == nil {
			skipped = append(skipped, anthropicBuiltinToolType(&tools[i]))
			continue
		}
		if t.Name == "" {
			return nil, nil, fmt.Errorf("%w: tools[%d] has no name", internalapi.ErrInvalidRequestBody, i)
		}
		decl := &genai.FunctionDeclaration{Name: t.Name, Description: t.Description}
		if len(t.InputSchema) > 0 {
			var schema map[string]any
			if err = json.Unmarshal(t.InputSchema, &schema); err != nil {
				return nil, nil, fmt.Errorf("%w: input_schema of tool %s must be a JSON object", internalapi.ErrInvalidRequestBody, t.Name)
			}
			if len(schema) > 0 {
				if parametersJSONSchemaAvailable {
					decl.ParametersJsonSchema = schema
				} else if decl.Parameters, err = jsonSchemaToGemini(schema); err != nil {
					return nil, nil, fmt.Errorf("invalid input_schema in tool %s: %w", t.Name, err)
				}
			}
		}
		decls = append(decls, decl)
	}
	if len(decls) == 0 {
		return nil, skipped, nil
	}
	return []genai.Tool{{FunctionDeclarations: decls}}, skipped, nil
}

// anthropicBuiltinToolType returns the type of a built-in tool; tool types unknown to the unmarshaler
// are reported as "unknown".
func anthropicBuiltinToolType(t *anthropic.ToolUnion) string {
	switch {
	case t.BashTool != nil:
		return t.BashTool.Type
	case t.TextEditorTool20250124 != nil:
		return t.TextEditorTool20250124.Type
	case t.TextEditorTool20250429 != nil:
		return t.TextEditorTool20250429.Type
	case t.TextEditorTool20250728 != nil:
		return t.TextEditorTool20250728.Type
	case t.WebSearchTool != nil:
		return t.WebSearchTool.Type
	}
	return "unknown"
}

// anthropicToolChoiceToGemini maps tool_choice to the Gemini function calling mode.
// disable_parallel_tool_use has no Gemini equivalent and is ignored.
func anthropicToolChoiceToGemini(tc *anthropic.ToolChoice) *genai.ToolConfig {
	if tc == nil {
		return nil
	}
	cfg := &genai.FunctionCallingConfig{}
	switch {
	case tc.Auto != nil:
		cfg.Mode = genai.FunctionCallingConfigModeAuto
	case tc.Any != nil:
		cfg.Mode = genai.FunctionCallingConfigModeAny
	case tc.Tool != nil:
		cfg.Mode = genai.FunctionCallingConfigModeAny
		cfg.AllowedFunctionNames = []string{tc.Tool.Name}
	case tc.None != nil:
		cfg.Mode = genai.FunctionCallingConfigModeNone
	default:
		return nil
	}
	return &genai.ToolConfig{FunctionCallingConfig: cfg}
}

// anthropicToGeminiGenerationConfig maps sampling parameters, thinking and output_config. A json_schema
// output format is mapped the same way as the OpenAI response_format in openAIReqToGeminiGenerationConfig:
// the JSON schema is passed as is on models accepting it and converted to the Gemini schema otherwise.
func anthropicToGeminiGenerationConfig(
	body *anthropic.MessagesRequest, model internalapi.RequestModel, outputConfig anthropicOutputConfig,
) (*genai.GenerationConfig, error) {
	gc := &genai.GenerationConfig{
		StopSequences:  body.StopSequences,
		ThinkingConfig: anthropicThinkingToGemini(body.Thinking, outputConfig.effort, model),
	}
	if body.MaxTokens > 0 {
		gc.MaxOutputTokens = int32(min(body.MaxTokens, math.MaxInt32))
	}
	if body.Temperature != nil {
		gc.Temperature = ptr.To(float32(*body.Temperature))
	}
	if body.TopP != nil {
		gc.TopP = ptr.To(float32(*body.TopP))
	}
	if body.TopK != nil {
		gc.TopK = ptr.To(float32(*body.TopK))
	}
	if outputConfig.jsonSchema != nil {
		gc.ResponseMIMEType = mimeTypeApplicationJSON
		if responseJSONSchemaAvailable(model) {
			gc.ResponseJsonSchema = outputConfig.jsonSchema
		} else {
			schema, err := jsonSchemaToGemini(outputConfig.jsonSchema)
			if err != nil {
				return nil, fmt.Errorf("%w: invalid output_config.format.schema: %w", internalapi.ErrInvalidRequestBody, err)
			}
			gc.ResponseSchema = schema
		}
	}
	return gc, nil
}

// anthropicThinkingToGemini maps the Anthropic thinking configuration and output_config.effort. On models
// supporting thinking levels, a known effort decides the level unless thinking is disabled; otherwise the
// thinking setting alone is used.
func anthropicThinkingToGemini(t *anthropic.Thinking, effort string, model internalapi.RequestModel) *genai.ThinkingConfig {
	cfg := anthropicThinkingSettingToGemini(t, model)
	if t != nil && t.Disabled != nil {
		return cfg
	}
	level, ok := anthropicEffortToGeminiLevel(effort, model)
	if !ok {
		return cfg
	}
	if cfg == nil {
		cfg = &genai.ThinkingConfig{}
	}
	cfg.ThinkingLevel = level
	cfg.ThinkingBudget = nil
	return cfg
}

// anthropicEffortToGeminiLevel maps output_config.effort to a thinking level. high and the levels above it
// map to HIGH on every Gemini 3 model, since HIGH is the strongest level all of them accept; low and
// medium follow the OpenAI reasoning_effort rules. Unknown values and levels the model rejects are ignored.
func anthropicEffortToGeminiLevel(effort string, model internalapi.RequestModel) (genai.ThinkingLevel, bool) {
	if !reasoningEffortAvailable(model) {
		return "", false
	}
	var re openai.ReasoningEffort
	switch effort {
	case "low":
		re = openai.ReasoningEffortLow
	case "medium":
		re = openai.ReasoningEffortMedium
	case "high", "xhigh", "max":
		return genai.ThinkingLevelHigh, true
	default:
		return "", false
	}
	level, err := mapReasoningEffortToThinkingLevel(re, model)
	if err != nil {
		return "", false
	}
	return level, true
}

// anthropicThinkingSettingToGemini maps the Anthropic thinking setting. Gemini 3 models take a
// thinking level (and reject a budget in the same request), older models take a token budget:
// https://cloud.google.com/vertex-ai/generative-ai/docs/thinking
func anthropicThinkingSettingToGemini(t *anthropic.Thinking, model internalapi.RequestModel) *genai.ThinkingConfig {
	if t == nil {
		return nil
	}
	gemini3 := reasoningEffortAvailable(model)
	flash := isGeminiFlashModel(model)
	switch {
	case t.Enabled != nil:
		cfg := &genai.ThinkingConfig{IncludeThoughts: t.Enabled.Display != anthropicThinkingDisplayOmitted}
		if gemini3 {
			cfg.ThinkingLevel = anthropicThinkingBudgetToGeminiLevel(t.Enabled.BudgetTokens, flash)
		} else {
			cfg.ThinkingBudget = ptr.To(int32(min(t.Enabled.BudgetTokens, math.MaxInt32)))
		}
		return cfg
	case t.Adaptive != nil:
		cfg := &genai.ThinkingConfig{IncludeThoughts: t.Adaptive.Display != anthropicThinkingDisplayOmitted}
		if !gemini3 {
			cfg.ThinkingBudget = ptr.To(int32(-1)) // Dynamic thinking.
		}
		return cfg
	case t.Disabled != nil:
		// Gemini 3 cannot turn thinking off, and Vertex AI rejects THINKING_LEVEL_MINIMAL on some models,
		// so LOW is used for every Gemini 3 model.
		switch {
		case gemini3:
			return &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}
		case flash:
			return &genai.ThinkingConfig{ThinkingBudget: ptr.To(int32(0))}
		}
	}
	return nil
}

func anthropicThinkingBudgetToGeminiLevel(budget float64, flash bool) genai.ThinkingLevel {
	switch {
	case budget < anthropicThinkingBudgetLowMax:
		return genai.ThinkingLevelLow
	case budget < anthropicThinkingBudgetMediumMax && flash:
		return genai.ThinkingLevelMedium
	default:
		return genai.ThinkingLevelHigh
	}
}

// encodeGeminiThoughtSignature returns the bare prefix for an empty signature, so that thinking blocks
// without a Gemini signature can still be told apart from Anthropic-issued ones.
func encodeGeminiThoughtSignature(sig []byte) string {
	return geminiThoughtSignaturePrefix + base64.StdEncoding.EncodeToString(sig)
}

// decodeGeminiThoughtSignature returns nil for signatures that were not minted by this translator and
// for the bare prefix, which marks a thinking block without a Gemini signature.
func decodeGeminiThoughtSignature(s string) []byte {
	encoded, ok := strings.CutPrefix(s, geminiThoughtSignaturePrefix)
	if !ok {
		return nil
	}
	sig, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || len(sig) == 0 {
		return nil
	}
	return sig
}

// -------------------------------------------------------------
// Response conversion: Gemini -> Anthropic Messages
// -------------------------------------------------------------

// geminiResponseToAnthropic converts a non-streaming response. failure is non-empty when the finish reason
// reports a failed generation, in which case the response must be replaced with an api_error.
// Only the first candidate is used, since Anthropic responses have no candidates.
func geminiResponseToAnthropic(resp *genai.GenerateContentResponse, model string) (
	anthropicResp *anthropic.MessagesResponse, tokenUsage metrics.TokenUsage, failure string,
) {
	var parts []*genai.Part
	var finishReason genai.FinishReason
	var finishMessage string
	if len(resp.Candidates) > 0 && resp.Candidates[0] != nil {
		candidate := resp.Candidates[0]
		finishReason = candidate.FinishReason
		finishMessage = candidate.FinishMessage
		if candidate.Content != nil {
			parts = candidate.Content.Parts
		}
	}
	content, hasToolUse := geminiPartsToAnthropicContent(parts)
	usage, tokenUsage := geminiUsageToAnthropic(resp.UsageMetadata)
	stopReason, ok := geminiFinishReasonToAnthropic(finishReason, hasToolUse, geminiPromptBlocked(resp))
	if !ok {
		failure = geminiFinishFailureMessage(finishReason, finishMessage)
	}
	return &anthropic.MessagesResponse{
		ID:         anthropicMessageIDFromGemini(resp.ResponseID),
		Type:       anthropic.ConstantMessagesResponseTypeMessages("message"),
		Role:       anthropic.ConstantMessagesResponseRoleAssistant("assistant"),
		Content:    content,
		Model:      model,
		StopReason: &stopReason,
		Usage:      &usage,
	}, tokenUsage, failure
}

// geminiPartsToAnthropicContent converts Gemini parts to content blocks following the signature placement
// rule documented on [anthropicToGCPVertexAITranslator]. It also reports whether a tool_use was produced.
func geminiPartsToAnthropicContent(parts []*genai.Part) ([]anthropic.MessagesContentBlock, bool) {
	content := []anthropic.MessagesContentBlock{}
	var thinking strings.Builder
	var pendingSig []byte
	var openText *anthropic.TextBlock
	hasToolUse := false

	flushThinking := func(sig []byte) {
		if thinking.Len() == 0 && len(sig) == 0 {
			return
		}
		content = append(content, anthropic.MessagesContentBlock{Thinking: &anthropic.ThinkingBlock{
			Type:      "thinking",
			Thinking:  thinking.String(),
			Signature: encodeGeminiThoughtSignature(sig),
		}})
		thinking.Reset()
		openText = nil
	}
	takeSig := func(own []byte) []byte {
		sig := own
		if len(sig) == 0 {
			sig = pendingSig
		}
		pendingSig = nil
		return sig
	}

	for _, part := range parts {
		if part == nil {
			continue
		}
		switch {
		case part.Thought:
			thinking.WriteString(part.Text)
			if len(part.ThoughtSignature) > 0 {
				pendingSig = part.ThoughtSignature
			}
		case part.FunctionCall != nil:
			flushThinking(takeSig(part.ThoughtSignature))
			content = append(content, anthropic.MessagesContentBlock{Tool: geminiFunctionCallToToolUse(part.FunctionCall)})
			openText = nil
			hasToolUse = true
		case part.Text != "":
			flushThinking(nil)
			if len(part.ThoughtSignature) > 0 {
				pendingSig = part.ThoughtSignature
			}
			if openText == nil {
				openText = &anthropic.TextBlock{Type: "text"}
				content = append(content, anthropic.MessagesContentBlock{Text: openText})
			}
			openText.Text += part.Text
		case len(part.ThoughtSignature) > 0:
			// A signature on a part without data, e.g. the empty text part that closes a stream.
			pendingSig = part.ThoughtSignature
		}
	}
	// A signature not followed by a function call is dropped.
	flushThinking(nil)
	return content, hasToolUse
}

func geminiFunctionCallToToolUse(fc *genai.FunctionCall) *anthropic.ToolUseBlock {
	input := fc.Args
	if input == nil {
		input = map[string]any{}
	}
	return &anthropic.ToolUseBlock{Type: "tool_use", ID: geminiFunctionCallToolUseID(fc), Name: fc.Name, Input: input}
}

// geminiFunctionCallToolUseID keeps the Gemini function call id when it is a valid Anthropic tool_use id.
func geminiFunctionCallToolUseID(fc *genai.FunctionCall) string {
	if fc.ID != "" && anthropicToolUseIDPattern.MatchString(fc.ID) {
		return fc.ID
	}
	return "toolu_" + strings.ReplaceAll(uuid.NewString(), "-", "")
}

func anthropicMessageIDFromGemini(responseID string) string {
	return "msg_" + cmp.Or(responseID, strings.ReplaceAll(uuid.NewString(), "-", ""))
}

func geminiPromptBlocked(resp *genai.GenerateContentResponse) bool {
	return resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != ""
}

// geminiFinishReasonToAnthropic maps the finish reason. Gemini reports STOP for both natural stops and
// stop sequences, so stop_sequence is never produced. Blocked content maps to refusal, the Anthropic
// stop reason for policy interventions. The second result is false when the finish reason reports a
// failed generation (MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL, OTHER, LANGUAGE, NO_IMAGE and values
// unknown to this translator, the reasons [geminiFinishReasonToOpenAI] does not map to stop, length,
// content_filter or recitation). Such responses are returned as api_error, which clients like Claude
// Code retry, instead of a silently truncated message. An empty reason is accepted here; the streaming
// conversion treats a stream ending without one as a failure on its own.
func geminiFinishReasonToAnthropic(reason genai.FinishReason, hasToolUse, promptBlocked bool) (anthropic.StopReason, bool) {
	if promptBlocked {
		return anthropic.StopReasonRefusal, true
	}
	var stopReason anthropic.StopReason
	switch reason {
	case "", genai.FinishReasonStop:
		stopReason = anthropic.StopReasonEndTurn
	case genai.FinishReasonMaxTokens:
		stopReason = anthropic.StopReasonMaxTokens
	case genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent, genai.FinishReasonSPII, genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent, genai.FinishReasonImageRecitation:
		stopReason = anthropic.StopReasonRefusal
	default:
		return "", false
	}
	if hasToolUse {
		return anthropic.StopReasonToolUse, true
	}
	return stopReason, true
}

// geminiFinishFailureMessage builds the api_error message for a failed generation.
func geminiFinishFailureMessage(reason genai.FinishReason, finishMessage string) string {
	msg := "upstream generation failed with finish reason " + string(reason)
	if finishMessage != "" {
		msg += ": " + finishMessage
	}
	return msg
}

// anthropicAPIErrorBody returns an Anthropic error body of type api_error.
func anthropicAPIErrorBody(message string) anthropic.ErrorResponse {
	return anthropic.ErrorResponse{
		Type:  "error",
		Error: anthropic.ErrorResponseMessage{Type: "api_error", Message: message},
	}
}

// geminiUsageToAnthropic converts usage metadata. Gemini counts cached tokens within the prompt tokens
// while Anthropic reports them separately, and thinking tokens are billed as output tokens.
func geminiUsageToAnthropic(u *genai.GenerateContentResponseUsageMetadata) (anthropic.Usage, metrics.TokenUsage) {
	if u == nil {
		return anthropic.Usage{}, metrics.TokenUsage{}
	}
	cached := int64(u.CachedContentTokenCount)
	input := max(int64(u.PromptTokenCount)-cached, 0)
	output := int64(u.CandidatesTokenCount) + int64(u.ThoughtsTokenCount)
	tokenUsage := metrics.ExtractTokenUsageFromExplicitCaching(input, output, &cached, ptr.To(int64(0)))
	tokenUsage.SetReasoningTokens(uint32(u.ThoughtsTokenCount)) //nolint:gosec
	return anthropic.Usage{
		InputTokens:          float64(input),
		OutputTokens:         float64(output),
		CacheReadInputTokens: float64(cached),
	}, tokenUsage
}

// parseGCPVertexAIError parses a Vertex AI error body, which is an array for streaming requests.
func parseGCPVertexAIError(buf []byte) (*gcpVertexAIErrorDetails, bool) {
	trimmed := bytes.TrimSpace(buf)
	var gcpErr gcpVertexAIError
	if bytes.HasPrefix(trimmed, []byte("[")) {
		var gcpErrs []gcpVertexAIError
		if err := json.Unmarshal(trimmed, &gcpErrs); err != nil || len(gcpErrs) == 0 {
			return nil, false
		}
		gcpErr = gcpErrs[0]
	} else if err := json.Unmarshal(trimmed, &gcpErr); err != nil {
		return nil, false
	}
	if gcpErr.Error.Message == "" && gcpErr.Error.Status == "" && gcpErr.Error.Code == 0 {
		return nil, false
	}
	return &gcpErr.Error, true
}

func gcpVertexAIErrorMessage(e *gcpVertexAIErrorDetails) string {
	msg := e.Message
	if e.Status != "" {
		msg = e.Status + ": " + msg
	}
	if details := bytes.TrimSpace(e.Details); len(details) > 0 && !bytes.Equal(details, []byte("null")) {
		msg += "\nDetails: " + string(details)
	}
	return msg
}

// anthropicErrorTypeForHTTPStatus maps an HTTP status to the Anthropic error type:
// https://docs.claude.com/en/api/errors
func anthropicErrorTypeForHTTPStatus(status int) string {
	switch status {
	case 400:
		return "invalid_request_error"
	case 401:
		return "authentication_error"
	case 403:
		return "permission_error"
	case 404:
		return "not_found_error"
	case 413:
		return "request_too_large"
	case 429:
		return "rate_limit_error"
	case 503, 529:
		return "overloaded_error"
	case 504:
		return "timeout_error"
	}
	if status >= 400 && status < 500 {
		return "invalid_request_error"
	}
	return "api_error"
}

// -------------------------------------------------------------
// Streaming conversion: Gemini SSE -> Anthropic SSE
// -------------------------------------------------------------

// anthropicSSEUsageWithCache is the usage of message_start and message_delta events including cache
// fields, which [sseMessageUsage] and [sseOutputUsage] do not have.
type anthropicSSEUsageWithCache struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
}

type anthropicSSEMessageStartWithCache struct {
	Type    string                       `json:"type"`
	Message anthropicSSEMessageWithCache `json:"message"`
}

type anthropicSSEMessageWithCache struct {
	ID           string                     `json:"id"`
	Type         string                     `json:"type"`
	Role         string                     `json:"role"`
	Content      []any                      `json:"content"`
	Model        string                     `json:"model"`
	StopReason   any                        `json:"stop_reason"`
	StopSequence any                        `json:"stop_sequence"`
	Usage        anthropicSSEUsageWithCache `json:"usage"`
}

type anthropicSSEMessageDeltaWithCache struct {
	Type  string                     `json:"type"`
	Delta sseMessageDeltaBody        `json:"delta"`
	Usage anthropicSSEUsageWithCache `json:"usage"`
}

type geminiStreamBlock int

const (
	geminiStreamBlockNone geminiStreamBlock = iota
	geminiStreamBlockText
	geminiStreamBlockThinking
)

// geminiToAnthropicStreamState converts Gemini SSE chunks to Anthropic SSE events. It applies the same
// signature placement rule as [geminiPartsToAnthropicContent], so that streaming and non-streaming
// responses round-trip identically.
type geminiToAnthropicStreamState struct {
	requestModel string
	model        string
	messageID    string
	span         tracingapi.MessageSpan

	buffer    []byte
	delimiter []byte

	messageStarted bool
	finished       bool
	blockIndex     int
	openBlock      geminiStreamBlock
	// thinkingSigned reports whether the open thinking block already has a signature_delta.
	thinkingSigned bool
	pendingSig     []byte

	hasToolUse    bool
	promptBlocked bool
	finishReason  genai.FinishReason
	finishMessage string
	usage         anthropic.Usage
	tokenUsage    metrics.TokenUsage
}

func (s *geminiToAnthropicStreamState) process(data []byte, endOfStream bool, out *[]byte) error {
	s.buffer = append(s.buffer, data...)
	if s.delimiter == nil {
		s.delimiter = detectSSEDelimiter(s.buffer)
	}
	if s.delimiter != nil {
		for {
			event, rest, found := bytes.Cut(s.buffer, s.delimiter)
			if !found {
				break
			}
			s.buffer = rest
			if err := s.handleEvent(event, out); err != nil {
				return err
			}
		}
	}
	if !endOfStream {
		return nil
	}
	if len(bytes.TrimSpace(s.buffer)) > 0 {
		event := s.buffer
		s.buffer = nil
		if err := s.handleEvent(event, out); err != nil {
			return err
		}
	}
	return s.finish(out)
}

func (s *geminiToAnthropicStreamState) handleEvent(event []byte, out *[]byte) error {
	if s.finished {
		return nil
	}
	var payload []byte
	for _, line := range bytes.FieldsFunc(event, func(r rune) bool { return r == '\n' || r == '\r' }) {
		if data, ok := cutSSEDataPrefix(line); ok {
			if len(payload) > 0 {
				payload = append(payload, '\n')
			}
			payload = append(payload, data...)
		}
	}
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 {
		return nil
	}
	if gjson.GetBytes(payload, "error").Exists() {
		return s.emitError(payload, out)
	}
	var chunk genai.GenerateContentResponse
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return fmt.Errorf("failed to decode Gemini stream chunk: %w", err)
	}
	return s.handleChunk(&chunk, out)
}

func (s *geminiToAnthropicStreamState) handleChunk(chunk *genai.GenerateContentResponse, out *[]byte) error {
	if s.model == "" {
		s.model = chunk.ModelVersion
	}
	if s.messageID == "" && chunk.ResponseID != "" {
		s.messageID = anthropicMessageIDFromGemini(chunk.ResponseID)
	}
	if chunk.UsageMetadata != nil {
		s.usage, s.tokenUsage = geminiUsageToAnthropic(chunk.UsageMetadata)
	}
	if geminiPromptBlocked(chunk) {
		s.promptBlocked = true
	}
	if err := s.ensureMessageStarted(out); err != nil {
		return err
	}
	if len(chunk.Candidates) == 0 || chunk.Candidates[0] == nil {
		return nil
	}
	candidate := chunk.Candidates[0]
	if candidate.FinishReason != "" {
		s.finishReason = candidate.FinishReason
	}
	if candidate.FinishMessage != "" {
		s.finishMessage = candidate.FinishMessage
	}
	if candidate.Content == nil {
		return nil
	}
	for _, part := range candidate.Content.Parts {
		if err := s.handlePart(part, out); err != nil {
			return err
		}
	}
	return nil
}

func (s *geminiToAnthropicStreamState) handlePart(part *genai.Part, out *[]byte) error {
	if part == nil {
		return nil
	}
	switch {
	case part.Thought:
		if len(part.ThoughtSignature) > 0 {
			s.pendingSig = part.ThoughtSignature
		}
		if part.Text == "" {
			return nil
		}
		if err := s.openBlockOf(geminiStreamBlockThinking, out); err != nil {
			return err
		}
		return s.emit(out, "content_block_delta", sseContentBlockDeltaThinking{
			Type: "content_block_delta", Index: s.blockIndex,
			Delta: sseThinkingDelta{Type: "thinking_delta", Thinking: part.Text},
		})
	case part.FunctionCall != nil:
		if err := s.emitSignature(part.ThoughtSignature, out); err != nil {
			return err
		}
		if err := s.closeBlock(out); err != nil {
			return err
		}
		return s.emitToolUse(part.FunctionCall, out)
	case part.Text != "":
		// The signature is held for a following function call instead of splitting the text.
		if len(part.ThoughtSignature) > 0 {
			s.pendingSig = part.ThoughtSignature
		}
		if err := s.openBlockOf(geminiStreamBlockText, out); err != nil {
			return err
		}
		return s.emit(out, "content_block_delta", sseContentBlockDeltaText{
			Type: "content_block_delta", Index: s.blockIndex,
			Delta: sseTextDelta{Type: "text_delta", Text: part.Text},
		})
	case len(part.ThoughtSignature) > 0:
		s.pendingSig = part.ThoughtSignature
	}
	return nil
}

// emitSignature closes the reasoning that precedes the current part with a signature_delta. The part's
// own signature takes precedence over a pending one; a thinking block is opened when none is open.
func (s *geminiToAnthropicStreamState) emitSignature(own []byte, out *[]byte) error {
	sig := own
	if len(sig) == 0 {
		sig = s.pendingSig
	}
	s.pendingSig = nil
	if len(sig) == 0 {
		return nil
	}
	if err := s.openBlockOf(geminiStreamBlockThinking, out); err != nil {
		return err
	}
	if err := s.emit(out, "content_block_delta", sseContentBlockDeltaSignature{
		Type: "content_block_delta", Index: s.blockIndex,
		Delta: sseSignatureDelta{Type: "signature_delta", Signature: encodeGeminiThoughtSignature(sig)},
	}); err != nil {
		return err
	}
	s.thinkingSigned = true
	return s.closeBlock(out)
}

// openBlockOf ensures that a block of the given kind is open, closing a block of another kind first.
func (s *geminiToAnthropicStreamState) openBlockOf(kind geminiStreamBlock, out *[]byte) error {
	if s.openBlock == kind {
		return nil
	}
	if err := s.closeBlock(out); err != nil {
		return err
	}
	s.openBlock = kind
	if kind == geminiStreamBlockThinking {
		s.thinkingSigned = false
		return s.emit(out, "content_block_start", sseContentBlockStartThinking{
			Type: "content_block_start", Index: s.blockIndex,
			ContentBlock: sseThinkingInit{Type: "thinking", Thinking: ""},
		})
	}
	return s.emit(out, "content_block_start", sseContentBlockStartText{
		Type: "content_block_start", Index: s.blockIndex,
		ContentBlock: sseTextBlock{Type: "text", Text: ""},
	})
}

// closeBlock closes the open block. A thinking block closed without a Gemini signature gets the bare
// prefix as its signature, matching [geminiPartsToAnthropicContent].
func (s *geminiToAnthropicStreamState) closeBlock(out *[]byte) error {
	if s.openBlock == geminiStreamBlockNone {
		return nil
	}
	if s.openBlock == geminiStreamBlockThinking && !s.thinkingSigned {
		s.thinkingSigned = true
		if err := s.emit(out, "content_block_delta", sseContentBlockDeltaSignature{
			Type: "content_block_delta", Index: s.blockIndex,
			Delta: sseSignatureDelta{Type: "signature_delta", Signature: encodeGeminiThoughtSignature(nil)},
		}); err != nil {
			return err
		}
	}
	s.openBlock = geminiStreamBlockNone
	if err := s.emit(out, "content_block_stop", sseContentBlockStop{Type: "content_block_stop", Index: s.blockIndex}); err != nil {
		return err
	}
	s.blockIndex++
	return nil
}

// emitToolUse emits a complete tool_use block, since Gemini streams function calls in one piece.
func (s *geminiToAnthropicStreamState) emitToolUse(fc *genai.FunctionCall, out *[]byte) error {
	toolUse := geminiFunctionCallToToolUse(fc)
	args, err := json.Marshal(toolUse.Input)
	if err != nil {
		return fmt.Errorf("failed to marshal function call arguments: %w", err)
	}
	s.hasToolUse = true
	if err = s.emit(out, "content_block_start", sseContentBlockStartTool{
		Type: "content_block_start", Index: s.blockIndex,
		ContentBlock: sseToolBlock{Type: "tool_use", ID: toolUse.ID, Name: toolUse.Name, Input: map[string]any{}},
	}); err != nil {
		return err
	}
	if err = s.emit(out, "content_block_delta", sseContentBlockDeltaTool{
		Type: "content_block_delta", Index: s.blockIndex,
		Delta: sseInputJSONDelta{Type: "input_json_delta", PartialJSON: string(args)},
	}); err != nil {
		return err
	}
	if err = s.emit(out, "content_block_stop", sseContentBlockStop{Type: "content_block_stop", Index: s.blockIndex}); err != nil {
		return err
	}
	s.blockIndex++
	return nil
}

func (s *geminiToAnthropicStreamState) ensureMessageStarted(out *[]byte) error {
	if s.messageStarted {
		return nil
	}
	s.messageStarted = true
	if s.messageID == "" {
		s.messageID = anthropicMessageIDFromGemini("")
	}
	return s.emit(out, "message_start", anthropicSSEMessageStartWithCache{
		Type: "message_start",
		Message: anthropicSSEMessageWithCache{
			ID:      s.messageID,
			Type:    "message",
			Role:    "assistant",
			Content: []any{},
			Model:   cmp.Or(s.model, s.requestModel),
			Usage: anthropicSSEUsageWithCache{
				InputTokens:          int(s.usage.InputTokens),
				CacheReadInputTokens: int(s.usage.CacheReadInputTokens),
			},
		},
	})
}

// finish ends the message at the end of the stream. A stream that ended without a finish reason (e.g. a
// dropped upstream connection) or with a failure finish reason ends with an api_error event instead of
// message_delta and message_stop, so that clients do not take a truncated message as complete.
func (s *geminiToAnthropicStreamState) finish(out *[]byte) error {
	if s.finished {
		return nil
	}
	s.finished = true
	if err := s.ensureMessageStarted(out); err != nil {
		return err
	}
	// A signature not followed by a function call is dropped.
	s.pendingSig = nil
	if err := s.closeBlock(out); err != nil {
		return err
	}
	stopReason, ok := geminiFinishReasonToAnthropic(s.finishReason, s.hasToolUse, s.promptBlocked)
	switch {
	case s.finishReason == "" && !s.promptBlocked:
		return s.emit(out, "error", anthropicAPIErrorBody("upstream stream ended without finish reason"))
	case !ok:
		return s.emit(out, "error", anthropicAPIErrorBody(geminiFinishFailureMessage(s.finishReason, s.finishMessage)))
	}
	if err := s.emit(out, "message_delta", anthropicSSEMessageDeltaWithCache{
		Type:  "message_delta",
		Delta: sseMessageDeltaBody{StopReason: string(stopReason)},
		Usage: anthropicSSEUsageWithCache{
			InputTokens:          int(s.usage.InputTokens),
			CacheReadInputTokens: int(s.usage.CacheReadInputTokens),
			OutputTokens:         int(s.usage.OutputTokens),
		},
	}); err != nil {
		return err
	}
	return s.emit(out, "message_stop", sseMessageStop{Type: "message_stop"})
}

// emitError converts an error delivered in the middle of the stream to an Anthropic error event:
// https://docs.claude.com/en/docs/build-with-claude/streaming#error-events
func (s *geminiToAnthropicStreamState) emitError(payload []byte, out *[]byte) error {
	s.finished = true
	message := string(payload)
	status := 0
	if details, ok := parseGCPVertexAIError(payload); ok {
		message = gcpVertexAIErrorMessage(details)
		status = details.Code
	}
	return s.emit(out, "error", anthropic.ErrorResponse{
		Type:  "error",
		Error: anthropic.ErrorResponseMessage{Type: anthropicErrorTypeForHTTPStatus(status), Message: message},
	})
}

func (s *geminiToAnthropicStreamState) emit(out *[]byte, eventType string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal %s event: %w", eventType, err)
	}
	appendAnthropicSSEEvent(out, eventType, data)
	if s.span != nil {
		chunk := &anthropic.MessagesStreamChunk{}
		if err = json.Unmarshal(data, chunk); err == nil {
			s.span.RecordResponseChunk(chunk)
		}
	}
	return nil
}
