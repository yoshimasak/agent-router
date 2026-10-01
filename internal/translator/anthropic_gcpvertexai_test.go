// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
	"k8s.io/utils/ptr"

	"github.com/envoyproxy/ai-gateway/internal/apischema/anthropic"
	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

const (
	testGemini3Flash = "gemini-3.8-flash"
	testGemini3Pro   = "gemini-3.1-pro"
)

// recordingMessageSpan records both full responses and stream chunks.
type recordingMessageSpan struct {
	responses []*anthropic.MessagesResponse
	chunks    []*anthropic.MessagesStreamChunk
}

func (r *recordingMessageSpan) RecordResponse(resp *anthropic.MessagesResponse) {
	r.responses = append(r.responses, resp)
}

func (r *recordingMessageSpan) RecordResponseChunk(chunk *anthropic.MessagesStreamChunk) {
	r.chunks = append(r.chunks, chunk)
}
func (r *recordingMessageSpan) EndSpanOnError(int, []byte) {}
func (r *recordingMessageSpan) EndSpan()                   {}

func geminiSig(s string) string { return encodeGeminiThoughtSignature([]byte(s)) }

func b64(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func parseAnthropicRequest(t *testing.T, body string) *anthropic.MessagesRequest {
	t.Helper()
	var req anthropic.MessagesRequest
	require.NoError(t, json.Unmarshal([]byte(body), &req))
	return &req
}

func translateAnthropicRequest(t *testing.T, model, body string) *gcp.GenerateContentRequest {
	t.Helper()
	req, err := anthropicToGeminiRequest(parseAnthropicRequest(t, body), model, anthropicOutputConfig{}, nil)
	require.NoError(t, err)
	return req
}

func headerValue(headers []internalapi.Header, key string) string {
	for _, h := range headers {
		if h.Key() == key {
			return h.Value()
		}
	}
	return ""
}

func TestAnthropicToGCPVertexAI_RequestBody(t *testing.T) {
	for _, tc := range []struct {
		name         string
		override     string
		body         string
		expectedPath string
		expectStream bool
	}{
		{
			name:         "non-streaming",
			body:         `{"model":"gemini-3.8-flash","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`,
			expectedPath: "publishers/google/models/gemini-3.8-flash:generateContent",
		},
		{
			name:         "streaming",
			body:         `{"model":"gemini-3.8-flash","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"hi"}]}`,
			expectedPath: "publishers/google/models/gemini-3.8-flash:streamGenerateContent?alt=sse",
			expectStream: true,
		},
		{
			name:         "model name override",
			override:     "gemini-3.1-pro",
			body:         `{"model":"claude-opus","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`,
			expectedPath: "publishers/google/models/gemini-3.1-pro:generateContent",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := NewAnthropicToGCPVertexAITranslator(tc.override)
			headers, body, err := tr.RequestBody(nil, parseAnthropicRequest(t, tc.body), false)
			require.NoError(t, err)
			require.Equal(t, tc.expectedPath, headerValue(headers, pathHeaderName))
			require.Equal(t, strconv.Itoa(len(body)), headerValue(headers, contentLengthHeaderName))

			var gcpReq gcp.GenerateContentRequest
			require.NoError(t, json.Unmarshal(body, &gcpReq))
			require.Equal(t, []genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}}}}, gcpReq.Contents)

			respHeaders, err := tr.ResponseHeaders(nil)
			require.NoError(t, err)
			if tc.expectStream {
				require.Equal(t, eventStreamContentType, headerValue(respHeaders, contentTypeHeaderName))
			} else {
				require.Empty(t, respHeaders)
			}
		})
	}
}

func TestAnthropicToGeminiRequest_Contents(t *testing.T) {
	pngData := "png-bytes"
	pdfData := "%PDF-1.7"
	for _, tc := range []struct {
		name     string
		body     string
		expected []genai.Content
	}{
		{
			name: "text, image and document blocks",
			body: `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
				{"type":"text","text":"look"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + b64(pngData) + `"}},
				{"type":"image","source":{"type":"url","url":"https://example.com/cat.webp"}},
				{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + b64(pdfData) + `"}},
				{"type":"document","source":{"type":"text","media_type":"text/plain","data":"plain doc"}},
				{"type":"document","source":{"type":"url","url":"https://example.com/a.pdf"}},
				{"type":"document","source":{"type":"content","content":[{"type":"text","text":"inner"}]}},
				{"type":"search_result","source":"https://example.com","title":"Title","content":[{"type":"text","text":"snippet"}]}
			]}]}`,
			expected: []genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{
				{Text: "look"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte(pngData)}},
				{FileData: &genai.FileData{MIMEType: "image/webp", FileURI: "https://example.com/cat.webp"}},
				{InlineData: &genai.Blob{MIMEType: "application/pdf", Data: []byte(pdfData)}},
				{Text: "plain doc"},
				{FileData: &genai.FileData{MIMEType: "application/pdf", FileURI: "https://example.com/a.pdf"}},
				{Text: "inner"},
				{Text: "Title\nhttps://example.com\nsnippet"},
			}}},
		},
		{
			name: "tool result with image and PDF uses multimodal function response parts",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"read it"},
				{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"a.png"}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[
					{"type":"text","text":"file contents"},
					{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + b64(pngData) + `"}},
					{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + b64(pdfData) + `"}},
					{"type":"text","text":"more"}
				]}]}
			]}`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "read it"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{
					FunctionCall:     &genai.FunctionCall{ID: "toolu_1", Name: "Read", Args: map[string]any{"path": "a.png"}},
					ThoughtSignature: dummyThoughtSignature,
				}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					ID:       "toolu_1",
					Name:     "Read",
					Response: map[string]any{"output": "file contents\nmore"},
					Parts: []*genai.FunctionResponsePart{
						{InlineData: &genai.FunctionResponseBlob{MIMEType: "image/png", Data: []byte(pngData)}},
						{InlineData: &genai.FunctionResponseBlob{MIMEType: "application/pdf", Data: []byte(pdfData)}},
					},
				}}}},
			},
		},
		{
			name: "tool result error and string content, parallel results grouped",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[
					{"type":"tool_use","id":"a","name":"Bash","input":{}},
					{"type":"tool_use","id":"b","name":"Grep","input":{}}
				]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"a","content":"boom","is_error":true},
					{"type":"tool_result","tool_use_id":"b","content":[{"type":"image","source":{"type":"url","url":"https://example.com/x.png"}}]},
					{"type":"text","text":"continue"}
				]}
			]}`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "Bash", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
					{FunctionCall: &genai.FunctionCall{ID: "b", Name: "Grep", Args: map[string]any{}}},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "Bash", Response: map[string]any{"error": "boom"}}},
					{FunctionResponse: &genai.FunctionResponse{
						ID: "b", Name: "Grep", Response: map[string]any{"output": ""},
						Parts: []*genai.FunctionResponsePart{{FileData: &genai.FunctionResponseFileData{MIMEType: "image/png", FileURI: "https://example.com/x.png"}}},
					}},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "continue"}}},
			},
		},
		{
			name: "consecutive same-role messages are merged and empty messages dropped",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"a"},
				{"role":"user","content":[{"type":"text","text":"b"}]},
				{"role":"assistant","content":[{"type":"redacted_thinking","data":"x"}]},
				{"role":"user","content":"c"}
			]}`,
			expected: []genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "a"}, {Text: "b"}, {Text: "c"}}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := translateAnthropicRequest(t, testGemini3Flash, tc.body)
			require.Equal(t, tc.expected, req.Contents)
		})
	}
}

func TestAnthropicToGeminiRequest_Errors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		body   string
		errMsg string
	}{
		{
			name:   "unknown tool_use_id",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"nope","content":"x"}]}]}`,
			errMsg: `unknown tool_use_id "nope"`,
		},
		{
			name:   "invalid base64 image",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"***"}}]}]}`,
			errMsg: "invalid base64 image data",
		},
		{
			name:   "invalid base64 document",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"***"}}]}]}`,
			errMsg: "invalid base64 document data",
		},
		{
			name:   "invalid input schema",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"t","input_schema":[1]}]}`,
			errMsg: "input_schema of tool t must be a JSON object",
		},
		{
			name:   "Files API image source",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"file_1"}}]}]}`,
			errMsg: "unsupported image source type for GCPVertexAI backends (supported: base64, url)",
		},
		{
			name: "Files API image source inside a tool result",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"Read","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":[{"type":"image","source":{"type":"file","file_id":"file_1"}}]}]}
			]}`,
			errMsg: "unsupported image source type",
		},
		{
			name:   "Files API document source",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file","file_id":"file_1"}}]}]}`,
			errMsg: "unsupported document source type for GCPVertexAI backends (supported: base64, text, url, content)",
		},
		{
			name: "assistant prefill",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"answer in JSON"},
				{"role":"assistant","content":"{"}
			]}`,
			errMsg: "assistant prefill is not supported for GCPVertexAI backends",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := anthropicToGeminiRequest(parseAnthropicRequest(t, tc.body), testGemini3Flash, anthropicOutputConfig{}, nil)
			require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
			require.ErrorContains(t, err, tc.errMsg)
		})
	}

	t.Run("unsupported role", func(t *testing.T) {
		_, err := anthropicMessagesToGeminiContents([]anthropic.MessageParam{{Role: "developer", Content: anthropic.MessageContent{Text: "x"}}})
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
	})
}

func TestAnthropicToGeminiRequest_Validation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		body   string
		errMsg string
	}{
		{
			name:   "all messages empty",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":""},{"role":"user","content":[]}]}`,
			errMsg: "messages must contain at least one non-empty message",
		},
		{
			name: "trailing empty user message leaves a model turn last",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"hello"},
				{"role":"user","content":[{"type":"text","text":""}]}
			]}`,
			errMsg: "assistant prefill is not supported for GCPVertexAI backends",
		},
		{
			name: "trailing empty system message leaves a model turn last",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":"hello"},
				{"role":"system","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}
			]}`,
			errMsg: "assistant prefill is not supported for GCPVertexAI backends",
		},
		{
			name:   "empty image url",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":""}}]}]}`,
			errMsg: "image url must not be empty",
		},
		{
			name:   "empty base64 image data",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":""}}]}]}`,
			errMsg: "base64 image source requires media_type and data",
		},
		{
			name:   "missing image media_type",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","data":"` + b64("x") + `"}}]}]}`,
			errMsg: "base64 image source requires media_type and data",
		},
		{
			name:   "empty document url",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"url","url":""}}]}]}`,
			errMsg: "document url must not be empty",
		},
		{
			name:   "empty base64 document data",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":""}}]}]}`,
			errMsg: "base64 document source requires data",
		},
		{
			name: "empty image url inside a tool result",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"Read","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":[{"type":"image","source":{"type":"url","url":""}}]}]}
			]}`,
			errMsg: "image url must not be empty",
		},
		{
			name:   "custom tool without a name",
			body:   `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"Read"},{"name":"","input_schema":{"type":"object"}}]}`,
			errMsg: "tools[1] has no name",
		},
		{
			name: "tool_choice names an undeclared tool",
			body: `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
				"tools":[{"name":"Read"}],"tool_choice":{"type":"tool","name":"Write"}}`,
			errMsg: `tool_choice refers to tool "Write", which is not a custom tool in tools`,
		},
		{
			name: "tool_choice names a skipped built-in tool",
			body: `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
				"tools":[{"name":"Read"},{"type":"web_search_20250305","name":"web_search"}],"tool_choice":{"type":"tool","name":"web_search"}}`,
			errMsg: `tool_choice refers to tool "web_search"`,
		},
		{
			name: "tool_choice names a tool without any tools",
			body: `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
				"tool_choice":{"type":"tool","name":"Read"}}`,
			errMsg: `tool_choice refers to tool "Read"`,
		},
		{
			name: "tool_result before its tool_use",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"x"}]},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"Read","input":{}}]},
				{"role":"user","content":"next"}
			]}`,
			errMsg: `unknown tool_use_id "a"`,
		},
		{
			name: "tool_result of a tool_use without a name",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"x"}]}
			]}`,
			errMsg: `unknown tool_use_id "a"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := anthropicToGeminiRequest(parseAnthropicRequest(t, tc.body), testGemini3Flash, anthropicOutputConfig{}, nil)
			require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
			require.ErrorContains(t, err, tc.errMsg)
		})
	}

	t.Run("trailing non-empty system message is a user turn", func(t *testing.T) {
		req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","content":"hello"},
			{"role":"system","content":"# Environment"}
		]}`)
		require.Len(t, req.Contents, 3)
		require.Equal(t, genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "# Environment"}}}, req.Contents[2])
	})
}

func TestAnthropicToGeminiRequest_ThinkingOnlyAssistant(t *testing.T) {
	for _, tc := range []struct {
		name     string
		body     string
		expected []genai.Content
	}{
		{
			name: "thinking text with a Gemini signature is kept as a thought part",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"a"},
				{"role":"assistant","content":[{"type":"thinking","thinking":"let me see","signature":"` + geminiSig("sig") + `"}]},
				{"role":"user","content":"b"}
			]}`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "a"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "let me see", Thought: true, ThoughtSignature: []byte("sig")}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "b"}}},
			},
		},
		{
			name: "several thinking blocks are joined and a non-Gemini signature is dropped",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"a"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"first","signature":"` + b64("anthropic") + `"},
					{"type":"redacted_thinking","data":"x"},
					{"type":"thinking","thinking":"second","signature":"gemini:"}
				]},
				{"role":"user","content":"b"}
			]}`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "a"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "first\nsecond", Thought: true}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "b"}}},
			},
		},
		{
			name: "empty thinking text (display omitted) drops the turn since Vertex AI rejects a thought part without text",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"a"},
				{"role":"assistant","content":[{"type":"thinking","thinking":"","signature":"` + geminiSig("sig") + `"}]},
				{"role":"user","content":"b"}
			]}`,
			expected: []genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "a"}, {Text: "b"}}}},
		},
		{
			name: "thinking text next to a tool_use is still omitted",
			body: `{"model":"m","max_tokens":1,"messages":[
				{"role":"user","content":"a"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"let me see","signature":"` + geminiSig("sig") + `"},
					{"type":"tool_use","id":"t","name":"Read","input":{}}
				]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"x"}]}
			]}`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "a"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{ID: "t", Name: "Read", Args: map[string]any{}}, ThoughtSignature: []byte("sig"),
				}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "t", Name: "Read", Response: map[string]any{"output": "x"}}}}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := translateAnthropicRequest(t, testGemini3Flash, tc.body)
			require.Equal(t, tc.expected, req.Contents)
		})
	}
}

func TestParseAnthropicOutputConfig(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"title": map[string]any{"type": "string"}},
		"required":             []any{"title"},
		"additionalProperties": false,
	}
	for _, tc := range []struct {
		name     string
		raw      string
		expected anthropicOutputConfig
		errMsg   string
	}{
		{name: "nil body", raw: "", expected: anthropicOutputConfig{}},
		{name: "no output_config", raw: `{"model":"m"}`, expected: anthropicOutputConfig{}},
		{name: "effort only", raw: `{"output_config":{"effort":"low"}}`, expected: anthropicOutputConfig{effort: "low"}},
		{name: "null format", raw: `{"output_config":{"format":null}}`, expected: anthropicOutputConfig{}},
		{
			name:     "json_schema format with effort",
			raw:      `{"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}}}}`,
			expected: anthropicOutputConfig{effort: "high", jsonSchema: schema},
		},
		{name: "format is not an object", raw: `{"output_config":{"format":"json"}}`, errMsg: "output_config.format must be an object"},
		{name: "unknown format type", raw: `{"output_config":{"format":{"type":"text"}}}`, errMsg: `unsupported output_config.format.type "text" (supported: json_schema)`},
		{name: "missing schema", raw: `{"output_config":{"format":{"type":"json_schema"}}}`, errMsg: "output_config.format.schema must be a JSON object"},
		{name: "schema is not an object", raw: `{"output_config":{"format":{"type":"json_schema","schema":[1]}}}`, errMsg: "output_config.format.schema must be a JSON object"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			oc, err := parseAnthropicOutputConfig([]byte(tc.raw))
			if tc.errMsg != "" {
				require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
				require.ErrorContains(t, err, tc.errMsg)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expected, oc)
		})
	}
}

func TestAnthropicToGCPVertexAI_RequestBody_OutputFormat(t *testing.T) {
	// Tools and a json_schema format together: Vertex AI accepts both, so neither is dropped.
	raw := []byte(`{"model":"m","max_tokens":512,"messages":[{"role":"user","content":"summarize"}],
		"tools":[{"name":"Read","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
		"output_config":{"format":{"type":"json_schema","schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}}}}`)
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"title": map[string]any{"type": "string"}},
		"required":             []any{"title"},
		"additionalProperties": false,
	}

	t.Run("Gemini 3 passes the JSON schema as is and keeps the tools", func(t *testing.T) {
		tr := NewAnthropicToGCPVertexAITranslator(testGemini3Flash)
		_, body, err := tr.RequestBody(raw, parseAnthropicRequest(t, string(raw)), false)
		require.NoError(t, err)
		var gcpReq gcp.GenerateContentRequest
		require.NoError(t, json.Unmarshal(body, &gcpReq))
		require.Equal(t, &genai.GenerationConfig{
			MaxOutputTokens:    512,
			ResponseMIMEType:   mimeTypeApplicationJSON,
			ResponseJsonSchema: schema,
		}, gcpReq.GenerationConfig)
		require.Len(t, gcpReq.Tools, 1)
	})

	t.Run("older models use the converted Gemini schema", func(t *testing.T) {
		tr := NewAnthropicToGCPVertexAITranslator("gemini-2.0-flash")
		_, body, err := tr.RequestBody(raw, parseAnthropicRequest(t, string(raw)), false)
		require.NoError(t, err)
		var gcpReq gcp.GenerateContentRequest
		require.NoError(t, json.Unmarshal(body, &gcpReq))
		require.Equal(t, &genai.GenerationConfig{
			MaxOutputTokens:  512,
			ResponseMIMEType: mimeTypeApplicationJSON,
			ResponseSchema: &genai.Schema{
				Type:       genai.Type("object"),
				Properties: map[string]*genai.Schema{"title": {Type: genai.Type("string")}},
				Required:   []string{"title"},
			},
		}, gcpReq.GenerationConfig)
	})

	t.Run("effort and format together", func(t *testing.T) {
		raw := []byte(`{"model":"m","max_tokens":512,"messages":[{"role":"user","content":"x"}],
			"output_config":{"effort":"low","format":{"type":"json_schema","schema":{"type":"object"}}}}`)
		tr := NewAnthropicToGCPVertexAITranslator(testGemini3Flash)
		_, body, err := tr.RequestBody(raw, parseAnthropicRequest(t, string(raw)), false)
		require.NoError(t, err)
		var gcpReq gcp.GenerateContentRequest
		require.NoError(t, json.Unmarshal(body, &gcpReq))
		require.Equal(t, &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow}, gcpReq.GenerationConfig.ThinkingConfig)
		require.Equal(t, map[string]any{"type": "object"}, gcpReq.GenerationConfig.ResponseJsonSchema)
	})

	t.Run("no format leaves the response MIME type unset", func(t *testing.T) {
		raw := []byte(`{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],"output_config":{"effort":"low"}}`)
		tr := NewAnthropicToGCPVertexAITranslator(testGemini3Flash)
		_, body, err := tr.RequestBody(raw, parseAnthropicRequest(t, string(raw)), false)
		require.NoError(t, err)
		require.NotContains(t, string(body), "responseMimeType")
	})

	t.Run("unknown format type is rejected", func(t *testing.T) {
		raw := []byte(`{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],"output_config":{"format":{"type":"xml"}}}`)
		tr := NewAnthropicToGCPVertexAITranslator(testGemini3Flash)
		_, _, err := tr.RequestBody(raw, parseAnthropicRequest(t, string(raw)), false)
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
		require.ErrorContains(t, err, `unsupported output_config.format.type "xml"`)
	})

	t.Run("schema the Gemini schema cannot express is rejected on older models", func(t *testing.T) {
		raw := []byte(`{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
			"output_config":{"format":{"type":"json_schema","schema":{"type":"object","$ref":"#/$defs/missing"}}}}`)
		tr := NewAnthropicToGCPVertexAITranslator("gemini-2.0-flash")
		_, _, err := tr.RequestBody(raw, parseAnthropicRequest(t, string(raw)), false)
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
		require.ErrorContains(t, err, "invalid output_config.format.schema")
	})
}

func TestImageMIMETypeFromURL(t *testing.T) {
	for url, expected := range map[string]string{
		"https://example.com/cat.png":                          "image/png",
		"https://example.com/cat.webp?X-Goog-Signature=abc.de": "image/webp",
		"https://example.com/cat.gif#frame.1":                  "image/gif",
		"https://example.com/image?format=png":                 "image/jpeg",
		"https://example.com/":                                 "image/jpeg",
		"http://[::1/cat.png":                                  "image/jpeg", // Unparsable.
	} {
		require.Equal(t, expected, imageMIMETypeFromURL(url), url)
	}
}

func TestAnthropicToGeminiRequest_ThoughtSignatures(t *testing.T) {
	claudeSig := b64("claude-signature-bytes")
	for _, tc := range []struct {
		name     string
		messages string
		expected []genai.Content
	}{
		{
			name: "parallel function calls restore the signature on the first call only",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"plan","signature":"` + geminiSig("sig-1") + `"},
					{"type":"tool_use","id":"a","name":"A","input":{}},
					{"type":"tool_use","id":"b","name":"B","input":{}}
				]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"a","content":"1"},
					{"type":"tool_result","tool_use_id":"b","content":"2"}
				]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: []byte("sig-1")},
					{FunctionCall: &genai.FunctionCall{ID: "b", Name: "B", Args: map[string]any{}}},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "1"}}},
					{FunctionResponse: &genai.FunctionResponse{ID: "b", Name: "B", Response: map[string]any{"output": "2"}}},
				}},
			},
		},
		{
			name: "text-only answer with a trailing signature restores it on the last part",
			messages: `[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":[
					{"type":"text","text":"hello"},
					{"type":"text","text":"world"},
					{"type":"thinking","thinking":"","signature":"` + geminiSig("sig-end") + `"}
				]},
				{"role":"user","content":"again"}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "hello"}, {Text: "world", ThoughtSignature: []byte("sig-end")}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "again"}}},
			},
		},
		{
			name: "non-Gemini, malformed and redacted signatures are dropped",
			messages: `[
				{"role":"user","content":"hi"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"claude","signature":"` + claudeSig + `"},
					{"type":"text","text":"a"},
					{"type":"thinking","thinking":"bad","signature":"gemini:***"},
					{"type":"redacted_thinking","data":"opaque"},
					{"type":"text","text":"b"}
				]},
				{"role":"user","content":"next"}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "a"}, {Text: "b"}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "next"}}},
			},
		},
		{
			name: "dummy signature is added to the first call of each current-turn step only",
			messages: `[
				{"role":"user","content":"old turn"},
				{"role":"assistant","content":[{"type":"tool_use","id":"old","name":"A","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"old","content":"r"}]},
				{"role":"assistant","content":[{"type":"text","text":"done"}]},
				{"role":"user","content":"new turn"},
				{"role":"assistant","content":[
					{"type":"tool_use","id":"s1a","name":"A","input":{}},
					{"type":"tool_use","id":"s1b","name":"A","input":{}}
				]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"s1a","content":"r"},
					{"type":"tool_result","tool_use_id":"s1b","content":"r"}
				]},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"","signature":"` + claudeSig + `"},
					{"type":"text","text":"checking"},
					{"type":"tool_use","id":"s2","name":"A","input":{}}
				]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"s2","content":"r"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "old turn"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "old", Name: "A", Args: map[string]any{}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "old", Name: "A", Response: map[string]any{"output": "r"}}}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "done"}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "new turn"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "s1a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
					{FunctionCall: &genai.FunctionCall{ID: "s1b", Name: "A", Args: map[string]any{}}},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "s1a", Name: "A", Response: map[string]any{"output": "r"}}},
					{FunctionResponse: &genai.FunctionResponse{ID: "s1b", Name: "A", Response: map[string]any{"output": "r"}}},
				}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{Text: "checking"},
					{FunctionCall: &genai.FunctionCall{ID: "s2", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "s2", Name: "A", Response: map[string]any{"output": "r"}}}}},
			},
		},
		{
			name: "tool results mixed with text do not start a new turn",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"A","input":{}}]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"a","content":"r"},
					{"type":"text","text":"<system-reminder>keep going</system-reminder>"}
				]},
				{"role":"assistant","content":[{"type":"tool_use","id":"b","name":"A","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"r"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "r"}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "<system-reminder>keep going</system-reminder>"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "b", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "b", Name: "A", Response: map[string]any{"output": "r"}}}}},
			},
		},
		{
			name: "a real signature is kept and no dummy is added",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"","signature":"` + geminiSig("real") + `"},
					{"type":"tool_use","id":"a","name":"A","input":{"x":1}}
				]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{"x": float64(1)}}, ThoughtSignature: []byte("real")},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "ok"}}}}},
			},
		},
		{
			name: "bare gemini: signatures are ignored and do not suppress the dummy signature",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"plan","signature":"gemini:"},
					{"type":"text","text":"reading"},
					{"type":"tool_use","id":"a","name":"A","input":{}},
					{"type":"text","text":""}
				]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{Text: "reading"},
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "ok"}}}}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"messages":`+tc.messages+`}`)
			require.Equal(t, tc.expected, req.Contents)
		})
	}
}

func TestAnthropicToGeminiRequest_SystemMessages(t *testing.T) {
	for _, tc := range []struct {
		name     string
		messages string
		expected []genai.Content
	}{
		{
			name: "trailing system message after the user prompt",
			messages: `[
				{"role":"user","content":[{"type":"text","text":"hi"}]},
				{"role":"system","content":[{"type":"text","text":"# Environment\nWorking directory: /repo","cache_control":{"type":"ephemeral"}}]}
			]`,
			expected: []genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{
				{Text: "hi"},
				{Text: "# Environment\nWorking directory: /repo"},
			}}},
		},
		{
			name: "mid-conversation system message joins text blocks and merges with the next user message",
			messages: `[
				{"role":"user","content":"q1"},
				{"role":"assistant","content":[{"type":"text","text":"a1"}]},
				{"role":"system","content":[{"type":"text","text":"s1"},{"type":"text","text":"s2"}]},
				{"role":"user","content":"q2"}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "q1"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "a1"}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "s1\ns2"}, {Text: "q2"}}},
			},
		},
		{
			name: "system message right after tool_result is split into its own content in the current turn",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"Bash","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"ok"}]},
				{"role":"system","content":[{"type":"text","text":"# Environment"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "Bash", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "Bash", Response: map[string]any{"output": "ok"}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "# Environment"}}},
			},
		},
		{
			name: "system message before tool_result goes to the content after the function responses",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[
					{"type":"tool_use","id":"a","name":"A","input":{}},
					{"type":"tool_use","id":"b","name":"B","input":{}}
				]},
				{"role":"system","content":"env"},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"a","content":"1"},
					{"type":"tool_result","tool_use_id":"b","content":"2"},
					{"type":"text","text":"more"}
				]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
					{FunctionCall: &genai.FunctionCall{ID: "b", Name: "B", Args: map[string]any{}}},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "1"}}},
					{FunctionResponse: &genai.FunctionResponse{ID: "b", Name: "B", Response: map[string]any{"output": "2"}}},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "env"}, {Text: "more"}}},
			},
		},
		{
			name: "system-only content between model steps is not a turn boundary",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"A","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"1"}]},
				{"role":"assistant","content":[{"type":"text","text":"partial"}]},
				{"role":"system","content":"note"},
				{"role":"assistant","content":[{"type":"tool_use","id":"b","name":"A","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"2"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "1"}}}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "partial"}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "note"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "b", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "b", Name: "A", Response: map[string]any{"output": "2"}}}}},
			},
		},
		{
			name: "non-text system blocks are ignored and empty system messages are dropped",
			messages: `[
				{"role":"user","content":"hi"},
				{"role":"system","content":[{"type":"image","source":{"type":"url","url":"https://example.com/x.png"}}]},
				{"role":"system","content":[{"type":"image","source":{"type":"url","url":"https://example.com/y.png"}},{"type":"text","text":"plain"}]}
			]`,
			expected: []genai.Content{{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "hi"}, {Text: "plain"}}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"messages":`+tc.messages+`}`)
			require.Equal(t, tc.expected, req.Contents)
		})
	}
}

func TestAnthropicToGeminiRequest_FunctionResponseSplit(t *testing.T) {
	pngData := "png-bytes"
	for _, tc := range []struct {
		name     string
		messages string
		expected []genai.Content
	}{
		{
			name: "second Claude Code round ends with separate function response and text contents",
			messages: `[
				{"role":"user","content":[{"type":"text","text":"<system-reminder>ctx</system-reminder>"},{"type":"text","text":"write a file"}]},
				{"role":"system","content":"# Environment"},
				{"role":"assistant","content":[
					{"type":"thinking","thinking":"","signature":"` + geminiSig("sig-w") + `"},
					{"type":"tool_use","id":"toolu_w","name":"Write","input":{"file_path":"a.txt","content":"x"}}
				]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_w","content":"written"}]},
				{"role":"system","content":[{"type":"text","text":"<total_tokens>1000</total_tokens>"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{Text: "<system-reminder>ctx</system-reminder>"}, {Text: "write a file"}, {Text: "# Environment"},
				}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{
					FunctionCall:     &genai.FunctionCall{ID: "toolu_w", Name: "Write", Args: map[string]any{"file_path": "a.txt", "content": "x"}},
					ThoughtSignature: []byte("sig-w"),
				}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "toolu_w", Name: "Write", Response: map[string]any{"output": "written"}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "<total_tokens>1000</total_tokens>"}}},
			},
		},
		{
			name: "text after tool_result in the same user message",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"A","input":{}}]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"a","content":"r"},
					{"type":"text","text":"<system-reminder>after</system-reminder>"}
				]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "r"}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "<system-reminder>after</system-reminder>"}}},
			},
		},
		{
			name: "text before tool_result in the same user message",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"A","input":{}}]},
				{"role":"user","content":[
					{"type":"text","text":"<system-reminder>before</system-reminder>"},
					{"type":"tool_result","tool_use_id":"a","content":"r"}
				]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "r"}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "<system-reminder>before</system-reminder>"}}},
			},
		},
		{
			name: "images inside tool_result stay in the function response and top-level images are split off",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"Read","input":{}}]},
				{"role":"user","content":[
					{"type":"tool_result","tool_use_id":"a","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + b64(pngData) + `"}}]},
					{"type":"image","source":{"type":"url","url":"https://example.com/top.png"}},
					{"type":"text","text":"see image"}
				]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "Read", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					ID: "a", Name: "Read", Response: map[string]any{"output": ""},
					Parts: []*genai.FunctionResponsePart{{InlineData: &genai.FunctionResponseBlob{MIMEType: "image/png", Data: []byte(pngData)}}},
				}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{
					{FileData: &genai.FileData{MIMEType: "image/png", FileURI: "https://example.com/top.png"}},
					{Text: "see image"},
				}},
			},
		},
		{
			name: "split-off text contents do not start a new turn",
			messages: `[
				{"role":"user","content":"go"},
				{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"A","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"1"},{"type":"text","text":"r1"}]},
				{"role":"assistant","content":[{"type":"tool_use","id":"b","name":"A","input":{}}]},
				{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":"2"},{"type":"text","text":"r2"}]}
			]`,
			expected: []genai.Content{
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "go"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "a", Name: "A", Response: map[string]any{"output": "1"}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "r1"}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "b", Name: "A", Args: map[string]any{}}, ThoughtSignature: dummyThoughtSignature},
				}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "b", Name: "A", Response: map[string]any{"output": "2"}}}}},
				{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "r2"}}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"messages":`+tc.messages+`}`)
			require.Equal(t, tc.expected, req.Contents)
		})
	}
}

func TestAnthropicToGeminiRequest_SystemToolsAndConfig(t *testing.T) {
	t.Run("system prompt as string and blocks", func(t *testing.T) {
		req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"system":"be brief","messages":[{"role":"user","content":"x"}]}`)
		require.Equal(t, &genai.Content{Parts: []*genai.Part{{Text: "be brief"}}}, req.SystemInstruction)

		req = translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"system":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},{"type":"text","text":"b"}],"messages":[{"role":"user","content":"x"}]}`)
		require.Equal(t, &genai.Content{Parts: []*genai.Part{{Text: "a"}, {Text: "b"}}}, req.SystemInstruction)
	})

	t.Run("custom tools use parametersJsonSchema and built-in tools are skipped", func(t *testing.T) {
		req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
			"tools":[
				{"name":"Read","description":"read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}},
				{"type":"web_search_20250305","name":"web_search"},
				{"type":"bash_20250124","name":"bash"}
			],
			"tool_choice":{"type":"tool","name":"Read"}}`)
		require.Equal(t, []genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "Read",
			Description: "read a file",
			ParametersJsonSchema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"path": map[string]any{"type": "string"}},
				"required":             []any{"path"},
				"additionalProperties": false,
			},
		}}}}, req.Tools)
		require.Equal(t, &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeAny, AllowedFunctionNames: []string{"Read"},
		}}, req.ToolConfig)
	})

	t.Run("older models use the Gemini schema", func(t *testing.T) {
		req := translateAnthropicRequest(t, "gemini-2.0-flash", `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
			"tools":[{"name":"Read","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]}`)
		require.Len(t, req.Tools, 1)
		decl := req.Tools[0].FunctionDeclarations[0]
		require.Nil(t, decl.ParametersJsonSchema)
		require.NotNil(t, decl.Parameters)
		require.Equal(t, genai.Type("object"), decl.Parameters.Type)
		require.Equal(t, genai.Type("string"), decl.Parameters.Properties["path"].Type)
	})

	t.Run("only built-in tools is rejected instead of dropping every tool", func(t *testing.T) {
		for _, tools := range []string{
			`[{"type":"web_search_20250305","name":"web_search"}]`,
			`[{"type":"web_search_20250305","name":"web_search"},{"type":"bash_20250124","name":"bash"}]`,
		} {
			_, err := anthropicToGeminiRequest(parseAnthropicRequest(t, `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
				"tools":`+tools+`,"tool_choice":{"type":"any"}}`), testGemini3Flash, anthropicOutputConfig{}, nil)
			require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
			require.ErrorContains(t, err, "server tools such as web_search are not supported for GCPVertexAI backends")
			require.ErrorContains(t, err, "web_search_20250305")
		}
	})

	t.Run("skipped built-in tools are logged at debug level", func(t *testing.T) {
		var logs bytes.Buffer
		logger := slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
		req, err := anthropicToGeminiRequest(parseAnthropicRequest(t, `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
			"tools":[{"name":"Read","input_schema":{}},{"type":"text_editor_20250728","name":"str_replace_based_edit_tool"},{"type":"computer_20250124","name":"computer"},
				{"type":"text_editor_20250124","name":"str_replace_editor"},{"type":"text_editor_20250429","name":"str_replace_based_edit_tool"}]}`),
			testGemini3Flash, anthropicOutputConfig{}, logger)
		require.NoError(t, err)
		require.Len(t, req.Tools, 1)
		require.Contains(t, logs.String(), "skipping Anthropic built-in tools")
		for _, typ := range []string{"text_editor_20250728", "unknown", "text_editor_20250124", "text_editor_20250429"} {
			require.Contains(t, logs.String(), typ)
		}

		// A nil logger is allowed.
		_, err = anthropicToGeminiRequest(parseAnthropicRequest(t, `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
			"tools":[{"name":"Read"},{"type":"bash_20250124","name":"bash"}]}`), testGemini3Flash, anthropicOutputConfig{}, nil)
		require.NoError(t, err)
	})

	t.Run("no tools means no tool config", func(t *testing.T) {
		req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":1,"messages":[{"role":"user","content":"x"}],
			"tool_choice":{"type":"any"}}`)
		require.Nil(t, req.Tools)
		require.Nil(t, req.ToolConfig)
	})

	t.Run("generation config", func(t *testing.T) {
		req := translateAnthropicRequest(t, testGemini3Flash, `{"model":"m","max_tokens":2048,"temperature":0.5,"top_p":0.9,"top_k":40,
			"stop_sequences":["END"],"metadata":{"user_id":"u"},"context_management":{"edits":[]},
			"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`)
		require.Equal(t, &genai.GenerationConfig{
			MaxOutputTokens: 2048,
			Temperature:     ptr.To(float32(0.5)),
			TopP:            ptr.To(float32(0.9)),
			TopK:            ptr.To(float32(40)),
			StopSequences:   []string{"END"},
		}, req.GenerationConfig)
	})
}

func TestAnthropicToolChoiceToGemini(t *testing.T) {
	for _, tc := range []struct {
		name     string
		choice   *anthropic.ToolChoice
		expected *genai.ToolConfig
	}{
		{name: "nil", choice: nil, expected: nil},
		{name: "empty", choice: &anthropic.ToolChoice{}, expected: nil},
		{
			name:     "auto",
			choice:   &anthropic.ToolChoice{Auto: &anthropic.ToolChoiceAuto{Type: "auto", DisableParallelToolUse: ptr.To(true)}},
			expected: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAuto}},
		},
		{
			name:     "any",
			choice:   &anthropic.ToolChoice{Any: &anthropic.ToolChoiceAny{Type: "any"}},
			expected: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeAny}},
		},
		{
			name:   "tool",
			choice: &anthropic.ToolChoice{Tool: &anthropic.ToolChoiceTool{Type: "tool", Name: "Read"}},
			expected: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAny, AllowedFunctionNames: []string{"Read"},
			}},
		},
		{
			name:     "none",
			choice:   &anthropic.ToolChoice{None: &anthropic.ToolChoiceNone{Type: "none"}},
			expected: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeNone}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.expected, anthropicToolChoiceToGemini(tc.choice))
		})
	}
}

func TestAnthropicThinkingToGemini(t *testing.T) {
	enabled := func(budget float64, display string) *anthropic.Thinking {
		return &anthropic.Thinking{Enabled: &anthropic.ThinkingEnabled{Type: "enabled", BudgetTokens: budget, Display: display}}
	}
	adaptive := func(display string) *anthropic.Thinking {
		return &anthropic.Thinking{Adaptive: &anthropic.ThinkingAdaptive{Type: "adaptive", Display: display}}
	}
	disabled := &anthropic.Thinking{Disabled: &anthropic.ThinkingDisabled{Type: "disabled"}}
	for _, tc := range []struct {
		name     string
		thinking *anthropic.Thinking
		model    string
		expected *genai.ThinkingConfig
	}{
		{name: "unset", thinking: nil, model: testGemini3Flash, expected: nil},
		{
			name: "enabled low budget on Gemini 3", thinking: enabled(4000, ""), model: testGemini3Flash,
			expected: &genai.ThinkingConfig{IncludeThoughts: true, ThinkingLevel: genai.ThinkingLevelLow},
		},
		{
			name: "enabled medium budget on Gemini 3 Flash", thinking: enabled(10000, "summarized"), model: testGemini3Flash,
			expected: &genai.ThinkingConfig{IncludeThoughts: true, ThinkingLevel: genai.ThinkingLevelMedium},
		},
		{
			name: "enabled medium budget on Gemini 3 Pro", thinking: enabled(10000, ""), model: testGemini3Pro,
			expected: &genai.ThinkingConfig{IncludeThoughts: true, ThinkingLevel: genai.ThinkingLevelHigh},
		},
		{
			name: "enabled high budget with omitted display", thinking: enabled(31999, "omitted"), model: testGemini3Flash,
			expected: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh},
		},
		{
			name: "enabled on Gemini 2.5 uses a budget", thinking: enabled(4000, ""), model: "gemini-2.5-flash",
			expected: &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: ptr.To(int32(4000))},
		},
		{
			name: "adaptive on Gemini 3 keeps the default dynamic level", thinking: adaptive(""), model: testGemini3Pro,
			expected: &genai.ThinkingConfig{IncludeThoughts: true},
		},
		{
			name: "adaptive on Gemini 2.5 uses a dynamic budget", thinking: adaptive("omitted"), model: "gemini-2.5-pro",
			expected: &genai.ThinkingConfig{ThinkingBudget: ptr.To(int32(-1))},
		},
		{
			name: "disabled on Gemini 3 Flash uses LOW since Vertex AI rejects MINIMAL", thinking: disabled, model: testGemini3Flash,
			expected: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
		},
		{
			name: "disabled on Gemini 3 Pro", thinking: disabled, model: testGemini3Pro,
			expected: &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelLow},
		},
		{
			name: "disabled on Gemini 2.5 Flash", thinking: disabled, model: "gemini-2.5-flash",
			expected: &genai.ThinkingConfig{ThinkingBudget: ptr.To(int32(0))},
		},
		{name: "disabled on Gemini 2.5 Pro", thinking: disabled, model: "gemini-2.5-pro", expected: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.expected, anthropicThinkingToGemini(tc.thinking, "", tc.model))
		})
	}
}

func TestAnthropicThinkingToGemini_Effort(t *testing.T) {
	enabled := func(budget float64) *anthropic.Thinking {
		return &anthropic.Thinking{Enabled: &anthropic.ThinkingEnabled{Type: "enabled", BudgetTokens: budget}}
	}
	adaptive := func(display string) *anthropic.Thinking {
		return &anthropic.Thinking{Adaptive: &anthropic.ThinkingAdaptive{Type: "adaptive", Display: display}}
	}
	disabled := &anthropic.Thinking{Disabled: &anthropic.ThinkingDisabled{Type: "disabled"}}
	level := func(l genai.ThinkingLevel, includeThoughts bool) *genai.ThinkingConfig {
		return &genai.ThinkingConfig{ThinkingLevel: l, IncludeThoughts: includeThoughts}
	}
	for i, tc := range []struct {
		model    string
		thinking *anthropic.Thinking
		effort   string
		expected *genai.ThinkingConfig
	}{
		// Gemini 3 Flash supports every level.
		{testGemini3Flash, adaptive("omitted"), "low", level(genai.ThinkingLevelLow, false)},
		{testGemini3Flash, adaptive("omitted"), "medium", level(genai.ThinkingLevelMedium, false)},
		{testGemini3Flash, adaptive(""), "high", level(genai.ThinkingLevelHigh, true)},
		{testGemini3Flash, adaptive(""), "xhigh", level(genai.ThinkingLevelHigh, true)},
		{testGemini3Flash, adaptive(""), "max", level(genai.ThinkingLevelHigh, true)},
		{testGemini3Flash, enabled(4000), "high", level(genai.ThinkingLevelHigh, true)},
		{testGemini3Flash, enabled(31999), "low", level(genai.ThinkingLevelLow, true)},
		{testGemini3Flash, nil, "medium", level(genai.ThinkingLevelMedium, false)},
		{testGemini3Flash, disabled, "high", level(genai.ThinkingLevelLow, false)},
		{testGemini3Flash, adaptive(""), "unknown", &genai.ThinkingConfig{IncludeThoughts: true}},
		{testGemini3Flash, adaptive(""), "none", &genai.ThinkingConfig{IncludeThoughts: true}},
		{testGemini3Flash, enabled(10000), "", level(genai.ThinkingLevelMedium, true)},
		{testGemini3Flash, nil, "", nil},
		// Gemini 3 Pro follows mapReasoningEffortToThinkingLevel for low and medium (medium becomes HIGH),
		// while high and above map to HIGH directly since the OpenAI mapping rejects high on Pro.
		{testGemini3Pro, adaptive("omitted"), "low", level(genai.ThinkingLevelLow, false)},
		{testGemini3Pro, adaptive("omitted"), "medium", level(genai.ThinkingLevelHigh, false)},
		{testGemini3Pro, adaptive(""), "high", level(genai.ThinkingLevelHigh, true)},
		{testGemini3Pro, adaptive(""), "max", level(genai.ThinkingLevelHigh, true)},
		{testGemini3Pro, enabled(4000), "max", level(genai.ThinkingLevelHigh, true)},
		{testGemini3Pro, disabled, "medium", level(genai.ThinkingLevelLow, false)},
		// Gemini 2.5 has no thinking level, so effort is ignored.
		{"gemini-2.5-flash", adaptive(""), "high", &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: ptr.To(int32(-1))}},
		{"gemini-2.5-flash", disabled, "high", &genai.ThinkingConfig{ThinkingBudget: ptr.To(int32(0))}},
		{"gemini-2.5-flash", nil, "medium", nil},
		{"gemini-2.5-pro", enabled(4000), "low", &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: ptr.To(int32(4000))}},
	} {
		t.Run(fmt.Sprintf("%02d_%s_%s", i, tc.model, tc.effort), func(t *testing.T) {
			require.Equal(t, tc.expected, anthropicThinkingToGemini(tc.thinking, tc.effort, tc.model))
		})
	}
}

func TestAnthropicToGCPVertexAI_RequestBody_ClaudeCodeShape(t *testing.T) {
	// Same shape as a request captured from Claude Code v2.1.285.
	raw := []byte(`{
		"model":"claude-opus-5-5","max_tokens":32000,"stream":true,
		"system":[{"type":"text","text":"You are Claude Code.","cache_control":{"type":"ephemeral"}}],
		"messages":[
			{"role":"user","content":[{"type":"text","text":"list files"}]},
			{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"a.go"}]},
			{"role":"system","content":[{"type":"text","text":"# Environment\nPlatform: linux"}]}
		],
		"tools":[{"name":"Bash","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}],
		"thinking":{"type":"adaptive","display":"omitted"},
		"output_config":{"effort":"medium"},
		"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},
		"metadata":{"user_id":"u"},
		"safeguards":{"mode":"default"}
	}`)
	for _, tc := range []struct {
		model    string
		expected *genai.ThinkingConfig
	}{
		{testGemini3Flash, &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelMedium}},
		{testGemini3Pro, &genai.ThinkingConfig{ThinkingLevel: genai.ThinkingLevelHigh}},
		{"gemini-2.5-flash", &genai.ThinkingConfig{ThinkingBudget: ptr.To(int32(-1))}},
	} {
		t.Run(tc.model, func(t *testing.T) {
			tr := NewAnthropicToGCPVertexAITranslator(tc.model)
			_, body, err := tr.RequestBody(raw, parseAnthropicRequest(t, string(raw)), false)
			require.NoError(t, err)
			var gcpReq gcp.GenerateContentRequest
			require.NoError(t, json.Unmarshal(body, &gcpReq))
			require.Equal(t, tc.expected, gcpReq.GenerationConfig.ThinkingConfig)
			require.Len(t, gcpReq.Contents, 4)
			require.Equal(t, genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
				{FunctionResponse: &genai.FunctionResponse{ID: "toolu_1", Name: "Bash", Response: map[string]any{"output": "a.go"}}},
			}}, gcpReq.Contents[2])
			require.Equal(t, genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: "# Environment\nPlatform: linux"}}}, gcpReq.Contents[3])
		})
	}
}

func TestGeminiThoughtSignatureEncoding(t *testing.T) {
	encoded := encodeGeminiThoughtSignature([]byte{0x01, 0xff})
	require.Equal(t, "gemini:Af8=", encoded)
	require.Equal(t, []byte{0x01, 0xff}, decodeGeminiThoughtSignature(encoded))
	require.Equal(t, "gemini:", encodeGeminiThoughtSignature(nil))
	for _, s := range []string{"", "Af8=", "gemini:", "gemini:%%%", "claude:Af8="} {
		require.Nil(t, decodeGeminiThoughtSignature(s), s)
	}
}

func TestGeminiPartsToAnthropicContent(t *testing.T) {
	for _, tc := range []struct {
		name        string
		parts       []*genai.Part
		expected    []anthropic.MessagesContentBlock
		expectsTool bool
	}{
		{
			name: "parallel function calls carry the signature before the first tool_use",
			parts: []*genai.Part{
				{Text: "summary", Thought: true},
				{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{"x": "1"}}, ThoughtSignature: []byte("sig")},
				{FunctionCall: &genai.FunctionCall{ID: "b", Name: "B"}},
			},
			expected: []anthropic.MessagesContentBlock{
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "summary", Signature: geminiSig("sig")}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "a", Name: "A", Input: map[string]any{"x": "1"}}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "b", Name: "B", Input: map[string]any{}}},
			},
			expectsTool: true,
		},
		{
			name: "text-only answer drops the signature on the last part and keeps the text in one block",
			parts: []*genai.Part{
				{Text: "Hello "},
				{Text: "world", ThoughtSignature: []byte("sig")},
			},
			expected: []anthropic.MessagesContentBlock{
				{Text: &anthropic.TextBlock{Type: "text", Text: "Hello world"}},
			},
		},
		{
			name: "text-only answer drops the signature on an empty trailing part",
			parts: []*genai.Part{
				{Text: "Hello "},
				{Text: "world"},
				{ThoughtSignature: []byte("sig")},
			},
			expected: []anthropic.MessagesContentBlock{
				{Text: &anthropic.TextBlock{Type: "text", Text: "Hello world"}},
			},
		},
		{
			name: "thought summary stays before the answer with the bare prefix when no function call follows",
			parts: []*genai.Part{
				{Text: "think", Thought: true, ThoughtSignature: []byte("sig")},
				{Text: "answer"},
				{ThoughtSignature: []byte("sig-end")},
			},
			expected: []anthropic.MessagesContentBlock{
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "think", Signature: geminiThoughtSignaturePrefix}},
				{Text: &anthropic.TextBlock{Type: "text", Text: "answer"}},
			},
		},
		{
			name: "signature on a thought part goes before the following function call",
			parts: []*genai.Part{
				{Text: "think", Thought: true, ThoughtSignature: []byte("sig")},
				{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A"}},
			},
			expected: []anthropic.MessagesContentBlock{
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "think", Signature: geminiSig("sig")}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "a", Name: "A", Input: map[string]any{}}},
			},
			expectsTool: true,
		},
		{
			name: "text then signed function call keeps the thinking block right before the tool_use",
			parts: []*genai.Part{
				{Text: "let me check", ThoughtSignature: []byte("sig-text")},
				{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A"}, ThoughtSignature: []byte("sig-fc")},
			},
			expected: []anthropic.MessagesContentBlock{
				{Text: &anthropic.TextBlock{Type: "text", Text: "let me check"}},
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Signature: geminiSig("sig-fc")}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "a", Name: "A", Input: map[string]any{}}},
			},
			expectsTool: true,
		},
		{
			name: "a held text signature goes to a following unsigned function call",
			parts: []*genai.Part{
				{Text: "let me check", ThoughtSignature: []byte("sig-text")},
				{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A"}},
			},
			expected: []anthropic.MessagesContentBlock{
				{Text: &anthropic.TextBlock{Type: "text", Text: "let me check"}},
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Signature: geminiSig("sig-text")}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "a", Name: "A", Input: map[string]any{}}},
			},
			expectsTool: true,
		},
		{
			name:     "thought only",
			parts:    []*genai.Part{{Text: "think", Thought: true}, nil},
			expected: []anthropic.MessagesContentBlock{{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "think", Signature: geminiThoughtSignaturePrefix}}},
		},
		{name: "no parts", parts: nil, expected: []anthropic.MessagesContentBlock{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			content, hasTool := geminiPartsToAnthropicContent(tc.parts)
			require.Equal(t, tc.expected, content)
			require.Equal(t, tc.expectsTool, hasTool)
		})
	}

	t.Run("function call ids", func(t *testing.T) {
		require.Equal(t, "call_1-x", geminiFunctionCallToolUseID(&genai.FunctionCall{ID: "call_1-x"}))
		for _, id := range []string{"", "has space", "a/b"} {
			generated := geminiFunctionCallToolUseID(&genai.FunctionCall{ID: id})
			require.True(t, strings.HasPrefix(generated, "toolu_"), generated)
			require.Regexp(t, `^[a-zA-Z0-9_-]+$`, generated)
		}
	})
}

func TestGeminiFinishReasonToAnthropic(t *testing.T) {
	for _, tc := range []struct {
		reason        genai.FinishReason
		hasToolUse    bool
		promptBlocked bool
		expected      anthropic.StopReason
		expectedOK    bool
	}{
		{reason: genai.FinishReasonStop, expected: anthropic.StopReasonEndTurn, expectedOK: true},
		{reason: genai.FinishReasonStop, hasToolUse: true, expected: anthropic.StopReasonToolUse, expectedOK: true},
		{reason: genai.FinishReasonMaxTokens, expected: anthropic.StopReasonMaxTokens, expectedOK: true},
		{reason: genai.FinishReasonMaxTokens, hasToolUse: true, expected: anthropic.StopReasonToolUse, expectedOK: true},
		{reason: genai.FinishReasonSafety, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: genai.FinishReasonRecitation, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: genai.FinishReasonProhibitedContent, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: genai.FinishReasonSPII, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: genai.FinishReasonBlocklist, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: genai.FinishReasonImageSafety, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: genai.FinishReasonImageRecitation, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: "", expected: anthropic.StopReasonEndTurn, expectedOK: true},
		{reason: "", promptBlocked: true, expected: anthropic.StopReasonRefusal, expectedOK: true},
		{reason: genai.FinishReasonOther, promptBlocked: true, expected: anthropic.StopReasonRefusal, expectedOK: true},
		// Failed generations, including the ones that came with a function call.
		{reason: genai.FinishReasonMalformedFunctionCall},
		{reason: genai.FinishReasonUnexpectedToolCall, hasToolUse: true},
		{reason: genai.FinishReasonOther},
		{reason: genai.FinishReasonLanguage},
		{reason: genai.FinishReasonNoImage},
		{reason: genai.FinishReasonImageOther},
		{reason: genai.FinishReasonUnspecified},
		{reason: "SOMETHING_NEW"},
	} {
		t.Run(fmt.Sprintf("%s/tool=%t/blocked=%t", tc.reason, tc.hasToolUse, tc.promptBlocked), func(t *testing.T) {
			stopReason, ok := geminiFinishReasonToAnthropic(tc.reason, tc.hasToolUse, tc.promptBlocked)
			require.Equal(t, tc.expectedOK, ok)
			require.Equal(t, tc.expected, stopReason)
		})
	}

	require.Equal(t, "upstream generation failed with finish reason MALFORMED_FUNCTION_CALL: Malformed function call: x(",
		geminiFinishFailureMessage(genai.FinishReasonMalformedFunctionCall, "Malformed function call: x("))
	require.Equal(t, "upstream generation failed with finish reason OTHER", geminiFinishFailureMessage(genai.FinishReasonOther, ""))
}

func TestGeminiUsageToAnthropic(t *testing.T) {
	usage, tokenUsage := geminiUsageToAnthropic(&genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        100,
		CachedContentTokenCount: 30,
		CandidatesTokenCount:    20,
		ThoughtsTokenCount:      10,
		TotalTokenCount:         130,
	})
	require.Equal(t, anthropic.Usage{InputTokens: 70, CacheReadInputTokens: 30, OutputTokens: 30}, usage)
	in, _ := tokenUsage.InputTokens()
	out, _ := tokenUsage.OutputTokens()
	total, _ := tokenUsage.TotalTokens()
	cached, _ := tokenUsage.CachedInputTokens()
	reasoning, ok := tokenUsage.ReasoningTokens()
	require.Equal(t, uint32(100), in)
	require.Equal(t, uint32(30), out)
	require.Equal(t, uint32(130), total)
	require.Equal(t, uint32(30), cached)
	require.True(t, ok)
	require.Equal(t, uint32(10), reasoning)

	usage, _ = geminiUsageToAnthropic(nil)
	require.Equal(t, anthropic.Usage{}, usage)
}

func TestAnthropicToGCPVertexAI_ResponseBody_NonStreaming(t *testing.T) {
	gcpResp := `{
		"candidates":[{"content":{"role":"model","parts":[
			{"text":"let me check","thought":true},
			{"text":"Checking."},
			{"functionCall":{"name":"Read","args":{"path":"a"}},"thoughtSignature":"` + b64("sig") + `"}
		]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":50,"cachedContentTokenCount":10,"candidatesTokenCount":7,"thoughtsTokenCount":3},
		"modelVersion":"gemini-3.8-flash-001",
		"responseId":"resp123"
	}`
	tr := NewAnthropicToGCPVertexAITranslator("")
	_, _, err := tr.RequestBody(nil, parseAnthropicRequest(t, `{"model":"gemini-3.8-flash","max_tokens":10,"messages":[{"role":"user","content":"x"}]}`), false)
	require.NoError(t, err)
	tr.(AnthropicResponseRedactor).SetRedactionConfig(true, true, slog.New(slog.DiscardHandler))

	span := &recordingMessageSpan{}
	headers, body, tokenUsage, model, err := tr.ResponseBody(nil, strings.NewReader(gcpResp), true, span)
	require.NoError(t, err)
	require.Equal(t, "gemini-3.8-flash-001", model)
	require.Equal(t, strconv.Itoa(len(body)), headerValue(headers, contentLengthHeaderName))
	in, _ := tokenUsage.InputTokens()
	require.Equal(t, uint32(50), in)

	var resp anthropic.MessagesResponse
	require.NoError(t, json.Unmarshal(body, &resp))
	require.Equal(t, "msg_resp123", resp.ID)
	require.Equal(t, "gemini-3.8-flash-001", resp.Model)
	require.Equal(t, anthropic.StopReasonToolUse, *resp.StopReason)
	require.Equal(t, &anthropic.Usage{InputTokens: 40, CacheReadInputTokens: 10, OutputTokens: 10}, resp.Usage)
	require.Equal(t, []anthropic.MessagesContentBlock{
		{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "let me check", Signature: geminiThoughtSignaturePrefix}},
		{Text: &anthropic.TextBlock{Type: "text", Text: "Checking."}},
		{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Signature: geminiSig("sig")}},
		{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: resp.Content[3].Tool.ID, Name: "Read", Input: map[string]any{"path": "a"}}},
	}, resp.Content)
	require.Regexp(t, `^toolu_[a-zA-Z0-9]+$`, resp.Content[3].Tool.ID)
	require.Len(t, span.responses, 1)
	require.Equal(t, resp.Content, span.responses[0].Content)
}

func newNonStreamingGCPVertexAITranslator(t *testing.T) AnthropicMessagesTranslator {
	t.Helper()
	tr := NewAnthropicToGCPVertexAITranslator("")
	_, _, err := tr.RequestBody(nil, parseAnthropicRequest(t, `{"model":"gemini-3.8-flash","max_tokens":10,"messages":[{"role":"user","content":"x"}]}`), false)
	require.NoError(t, err)
	return tr
}

func TestAnthropicToGCPVertexAI_ResponseBody_NonStreamingFailures(t *testing.T) {
	for _, tc := range []struct {
		name            string
		gcpResp         string
		expectedMessage string
	}{
		{
			name: "malformed function call with a finish message",
			gcpResp: `{"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]},
				"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"Malformed function call: Read("}],
				"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}`,
			expectedMessage: "upstream generation failed with finish reason MALFORMED_FUNCTION_CALL: Malformed function call: Read(",
		},
		{
			name:            "unexpected tool call without a finish message",
			gcpResp:         `{"candidates":[{"finishReason":"UNEXPECTED_TOOL_CALL"}]}`,
			expectedMessage: "upstream generation failed with finish reason UNEXPECTED_TOOL_CALL",
		},
		{
			name:            "other",
			gcpResp:         `{"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]},"finishReason":"OTHER"}]}`,
			expectedMessage: "upstream generation failed with finish reason OTHER",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := newNonStreamingGCPVertexAITranslator(t)
			span := &recordingMessageSpan{}
			headers, body, tokenUsage, _, err := tr.ResponseBody(nil, strings.NewReader(tc.gcpResp), true, span)
			require.NoError(t, err)
			require.Equal(t, "500", headerValue(headers, statusHeaderName))
			require.Contains(t, headers, internalapi.Header{contentTypeHeaderName, jsonContentType})
			require.Equal(t, strconv.Itoa(len(body)), headerValue(headers, contentLengthHeaderName))
			require.JSONEq(t, `{"type":"error","request_id":"","error":{"type":"api_error","message":`+strconv.Quote(tc.expectedMessage)+`}}`, string(body))
			require.Empty(t, span.responses)
			if strings.Contains(tc.gcpResp, "usageMetadata") {
				in, _ := tokenUsage.InputTokens()
				require.Equal(t, uint32(5), in)
			}
		})
	}
}

func TestAnthropicToGCPVertexAI_ResponseBody_NonStreamingMultipleCandidates(t *testing.T) {
	// Only the first candidate is used; candidateCount is never set by the translator.
	tr := newNonStreamingGCPVertexAITranslator(t)
	_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(`{"candidates":[
		{"index":0,"content":{"role":"model","parts":[{"text":"first"}]},"finishReason":"STOP"},
		{"index":1,"content":{"role":"model","parts":[{"text":"second"}]},"finishReason":"MALFORMED_FUNCTION_CALL"}
	]}`), true, nil)
	require.NoError(t, err)
	var resp anthropic.MessagesResponse
	require.NoError(t, json.Unmarshal(body, &resp))
	require.Equal(t, []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "first"}}}, resp.Content)
	require.Equal(t, anthropic.StopReasonEndTurn, *resp.StopReason)
}

// anthropicStreamResult is the Anthropic message reassembled from SSE events.
type anthropicStreamResult struct {
	events     []string
	id         string
	model      string
	startUsage *anthropic.Usage
	content    []anthropic.MessagesContentBlock
	stopReason anthropic.StopReason
	usage      anthropic.Usage
	errorBody  []byte
}

func accumulateAnthropicSSE(t *testing.T, sse []byte) anthropicStreamResult {
	t.Helper()
	var r anthropicStreamResult
	partialJSON := map[int]string{}
	for _, ev := range bytes.Split(bytes.TrimSpace(sse), []byte("\n\n")) {
		lines := bytes.SplitN(ev, []byte("\n"), 2)
		require.Len(t, lines, 2, string(ev))
		eventType := strings.TrimPrefix(string(lines[0]), "event: ")
		data := bytes.TrimPrefix(lines[1], []byte("data: "))
		r.events = append(r.events, eventType)
		if eventType == "error" {
			r.errorBody = data
			continue
		}
		var chunk anthropic.MessagesStreamChunk
		require.NoError(t, json.Unmarshal(data, &chunk))
		require.Equal(t, eventType, string(chunk.Type))
		switch {
		case chunk.MessageStart != nil:
			r.id, r.model, r.startUsage = chunk.MessageStart.ID, chunk.MessageStart.Model, chunk.MessageStart.Usage
		case chunk.ContentBlockStart != nil:
			require.Equal(t, len(r.content), chunk.ContentBlockStart.Index)
			r.content = append(r.content, chunk.ContentBlockStart.ContentBlock)
		case chunk.ContentBlockDelta != nil:
			idx, d := chunk.ContentBlockDelta.Index, chunk.ContentBlockDelta.Delta
			b := &r.content[idx]
			switch d.Type {
			case "text_delta":
				b.Text.Text += d.Text
			case "thinking_delta":
				b.Thinking.Thinking += d.Thinking
			case "signature_delta":
				b.Thinking.Signature = d.Signature
			case "input_json_delta":
				partialJSON[idx] += d.PartialJSON
			}
		case chunk.ContentBlockStop != nil:
			idx := chunk.ContentBlockStop.Index
			require.Equal(t, len(r.content)-1, idx)
			if js, ok := partialJSON[idx]; ok {
				var input map[string]any
				require.NoError(t, json.Unmarshal([]byte(js), &input))
				r.content[idx].Tool.Input = input
			}
		case chunk.MessageDelta != nil:
			r.stopReason, r.usage = chunk.MessageDelta.Delta.StopReason, chunk.MessageDelta.Usage
		}
	}
	return r
}

// geminiSSE builds a Gemini SSE stream where each element of partsPerChunk becomes one chunk.
// The last chunk also carries the finish reason and usage.
func geminiSSE(t *testing.T, delimiter string, partsPerChunk ...[]*genai.Part) []byte {
	t.Helper()
	var buf bytes.Buffer
	for i, parts := range partsPerChunk {
		candidate := map[string]any{"content": map[string]any{"role": "model", "parts": parts}}
		chunk := map[string]any{
			"candidates":    []any{candidate},
			"modelVersion":  "gemini-3.8-flash-001",
			"responseId":    "resp1",
			"usageMetadata": map[string]any{"promptTokenCount": 40},
		}
		if i == len(partsPerChunk)-1 {
			candidate["finishReason"] = "STOP"
			chunk["usageMetadata"] = map[string]any{
				"promptTokenCount": 40, "cachedContentTokenCount": 15, "candidatesTokenCount": 8, "thoughtsTokenCount": 4,
			}
		}
		data, err := json.Marshal(chunk)
		require.NoError(t, err)
		buf.WriteString("data: ")
		buf.Write(data)
		buf.WriteString(delimiter)
	}
	return buf.Bytes()
}

func newStreamingGCPVertexAITranslator(t *testing.T) AnthropicMessagesTranslator {
	t.Helper()
	tr := NewAnthropicToGCPVertexAITranslator("")
	_, _, err := tr.RequestBody(nil, parseAnthropicRequest(t, `{"model":"gemini-3.8-flash","max_tokens":10,"stream":true,"messages":[{"role":"user","content":"x"}]}`), false)
	require.NoError(t, err)
	return tr
}

// streamThrough feeds the stream to a new translator in pieces of the given size (0 means all at once).
func streamThrough(t *testing.T, sse []byte, pieceSize int, span *recordingMessageSpan) ([]byte, metrics.TokenUsage, string) {
	t.Helper()
	tr := newStreamingGCPVertexAITranslator(t)
	var out []byte
	var usage metrics.TokenUsage
	var model string
	feed := func(piece []byte, end bool) {
		var s tracingapi.MessageSpan
		if span != nil {
			s = span
		}
		_, body, tu, m, err := tr.ResponseBody(nil, bytes.NewReader(piece), end, s)
		require.NoError(t, err)
		require.NotNil(t, body)
		out = append(out, body...)
		usage, model = tu, m
	}
	if pieceSize == 0 {
		feed(sse, true)
		return out, usage, model
	}
	for i := 0; i < len(sse); i += pieceSize {
		feed(sse[i:min(i+pieceSize, len(sse))], false)
	}
	feed(nil, true)
	return out, usage, model
}

func TestAnthropicToGCPVertexAI_ResponseBody_Streaming(t *testing.T) {
	sse := geminiSSE(t, "\r\n\r\n",
		[]*genai.Part{{Text: "planning", Thought: true}},
		[]*genai.Part{{Text: "Let me "}},
		[]*genai.Part{{Text: "read."}},
		[]*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "Read", Args: map[string]any{"path": "a"}}, ThoughtSignature: []byte("sig")},
			{FunctionCall: &genai.FunctionCall{ID: "c2", Name: "Read", Args: map[string]any{"path": "b"}}},
		},
	)

	span := &recordingMessageSpan{}
	whole, tokenUsage, model := streamThrough(t, sse, 0, span)
	require.Equal(t, "gemini-3.8-flash-001", model)
	in, _ := tokenUsage.InputTokens()
	reasoning, _ := tokenUsage.ReasoningTokens()
	require.Equal(t, uint32(40), in)
	require.Equal(t, uint32(4), reasoning)

	r := accumulateAnthropicSSE(t, whole)
	require.Equal(t, []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", // thinking summary with the bare prefix
		"content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", // text
		"content_block_start", "content_block_delta", "content_block_stop", // signature carrier
		"content_block_start", "content_block_delta", "content_block_stop", // tool_use c1
		"content_block_start", "content_block_delta", "content_block_stop", // tool_use c2
		"message_delta", "message_stop",
	}, r.events)
	require.Equal(t, "msg_resp1", r.id)
	require.Equal(t, "gemini-3.8-flash-001", r.model)
	require.Equal(t, &anthropic.Usage{InputTokens: 40}, r.startUsage)
	require.Equal(t, anthropic.StopReasonToolUse, r.stopReason)
	require.Equal(t, anthropic.Usage{InputTokens: 25, CacheReadInputTokens: 15, OutputTokens: 12}, r.usage)
	require.Equal(t, []anthropic.MessagesContentBlock{
		{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "planning", Signature: geminiThoughtSignaturePrefix}},
		{Text: &anthropic.TextBlock{Type: "text", Text: "Let me read."}},
		{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Signature: geminiSig("sig")}},
		{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "c1", Name: "Read", Input: map[string]any{"path": "a"}}},
		{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "c2", Name: "Read", Input: map[string]any{"path": "b"}}},
	}, r.content)
	require.Len(t, span.chunks, len(r.events))

	for _, pieceSize := range []int{1, 7, 64} {
		split, _, _ := streamThrough(t, sse, pieceSize, nil)
		require.Equal(t, string(whole), string(split), "piece size %d", pieceSize)
	}

	lf, _, _ := streamThrough(t, bytes.ReplaceAll(sse, []byte("\r\n\r\n"), []byte("\n\n")), 5, nil)
	require.Equal(t, string(whole), string(lf))
}

func TestAnthropicToGCPVertexAI_ResponseBody_StreamingSignatures(t *testing.T) {
	for _, tc := range []struct {
		name     string
		chunks   [][]*genai.Part
		expected []anthropic.MessagesContentBlock
		stop     anthropic.StopReason
	}{
		{
			name:     "text-only answer drops the signature on the empty closing part",
			chunks:   [][]*genai.Part{{{Text: "Hello "}}, {{Text: "world"}}, {{Text: "", ThoughtSignature: []byte("sig")}}},
			expected: []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "Hello world"}}},
			stop:     anthropic.StopReasonEndTurn,
		},
		{
			name:     "text-only answer drops the signature on the last text part",
			chunks:   [][]*genai.Part{{{Text: "Hello "}}, {{Text: "world", ThoughtSignature: []byte("sig")}}},
			expected: []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "Hello world"}}},
			stop:     anthropic.StopReasonEndTurn,
		},
		{
			name: "thought summary then text then trailing signature keeps only the leading thinking block",
			chunks: [][]*genai.Part{
				{{Text: "hmm", Thought: true, ThoughtSignature: []byte("sig-thought")}},
				{{Text: "answer"}},
				{{Text: "", ThoughtSignature: []byte("sig-end")}},
			},
			expected: []anthropic.MessagesContentBlock{
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "hmm", Signature: geminiThoughtSignaturePrefix}},
				{Text: &anthropic.TextBlock{Type: "text", Text: "answer"}},
			},
			stop: anthropic.StopReasonEndTurn,
		},
		{
			name: "text then signed function call keeps the thinking block right before the tool_use",
			chunks: [][]*genai.Part{
				{{Text: "let me check"}},
				{{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: []byte("sig-fc")}},
			},
			expected: []anthropic.MessagesContentBlock{
				{Text: &anthropic.TextBlock{Type: "text", Text: "let me check"}},
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Signature: geminiSig("sig-fc")}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "a", Name: "A", Input: map[string]any{}}},
			},
			stop: anthropic.StopReasonToolUse,
		},
		{
			name: "signature in the middle of the text without a function call does not split the text",
			chunks: [][]*genai.Part{
				{{Text: "Hello "}},
				{{Text: "wor", ThoughtSignature: []byte("sig")}},
				{{Text: "ld"}},
			},
			expected: []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "Hello world"}}},
			stop:     anthropic.StopReasonEndTurn,
		},
		{
			name: "held text signature goes to a following unsigned function call",
			chunks: [][]*genai.Part{
				{{Text: "let me check", ThoughtSignature: []byte("sig-text")}},
				{{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}}},
			},
			expected: []anthropic.MessagesContentBlock{
				{Text: &anthropic.TextBlock{Type: "text", Text: "let me check"}},
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Signature: geminiSig("sig-text")}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "a", Name: "A", Input: map[string]any{}}},
			},
			stop: anthropic.StopReasonToolUse,
		},
		{
			name:     "pending signature after a thought is dropped at the end",
			chunks:   [][]*genai.Part{{{Text: "hmm", Thought: true}}, {{ThoughtSignature: []byte("sig")}}},
			expected: []anthropic.MessagesContentBlock{{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Thinking: "hmm", Signature: geminiThoughtSignaturePrefix}}},
			stop:     anthropic.StopReasonEndTurn,
		},
		{
			name: "signature-only thought part with empty text goes to the following function call",
			chunks: [][]*genai.Part{
				{{Text: "", Thought: true, ThoughtSignature: []byte("sig-thought")}},
				{{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}}},
			},
			expected: []anthropic.MessagesContentBlock{
				{Thinking: &anthropic.ThinkingBlock{Type: "thinking", Signature: geminiSig("sig-thought")}},
				{Tool: &anthropic.ToolUseBlock{Type: "tool_use", ID: "a", Name: "A", Input: map[string]any{}}},
			},
			stop: anthropic.StopReasonToolUse,
		},
		{
			name:     "nil parts are ignored",
			chunks:   [][]*genai.Part{{nil, {Text: "Hello"}}, {nil}},
			expected: []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "Hello"}}},
			stop:     anthropic.StopReasonEndTurn,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sse := geminiSSE(t, "\r\n\r\n", tc.chunks...)
			out, _, _ := streamThrough(t, sse, 0, nil)
			r := accumulateAnthropicSSE(t, out)
			require.Equal(t, tc.expected, r.content)
			require.Equal(t, tc.stop, r.stopReason)

			for _, pieceSize := range []int{1, 7, 64} {
				split, _, _ := streamThrough(t, sse, pieceSize, nil)
				require.Equal(t, string(out), string(split), "piece size %d", pieceSize)
			}

			var parts []*genai.Part
			for _, c := range tc.chunks {
				parts = append(parts, c...)
			}
			nonStreaming, _ := geminiPartsToAnthropicContent(parts)
			require.Equal(t, nonStreaming, r.content)
		})
	}
}

func TestAnthropicToGCPVertexAI_ResponseBody_StreamingEdgeCases(t *testing.T) {
	requireAPIError := func(t *testing.T, r anthropicStreamResult, message string) {
		t.Helper()
		require.Equal(t, "error", r.events[len(r.events)-1])
		require.NotContains(t, r.events, "message_delta")
		require.NotContains(t, r.events, "message_stop")
		require.JSONEq(t, `{"type":"error","request_id":"","error":{"type":"api_error","message":`+strconv.Quote(message)+`}}`, string(r.errorBody))
	}

	t.Run("empty stream ends with an api_error", func(t *testing.T) {
		out, _, model := streamThrough(t, nil, 0, nil)
		require.Equal(t, "gemini-3.8-flash", model)
		r := accumulateAnthropicSSE(t, out)
		require.Equal(t, []string{"message_start", "error"}, r.events)
		requireAPIError(t, r, "upstream stream ended without finish reason")
		require.Regexp(t, `^msg_[a-f0-9]+$`, r.id)
	})

	t.Run("stream ending without a finish reason ends with an api_error", func(t *testing.T) {
		// Written by hand since the JSON encoder used by geminiSSE does not sort map keys.
		sse := []byte(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}` + "\n\n")
		out, _, _ := streamThrough(t, sse, 0, nil)
		r := accumulateAnthropicSSE(t, out)
		require.Equal(t, []string{"message_start", "content_block_start", "content_block_delta", "content_block_stop", "error"}, r.events)
		requireAPIError(t, r, "upstream stream ended without finish reason")
	})

	t.Run("malformed function call ends with an api_error including the finish message", func(t *testing.T) {
		sse := []byte(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hmm","thought":true}]}}]}` + "\n\n" +
			`data: {"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"Malformed function call: Read("}],` +
			`"usageMetadata":{"promptTokenCount":9}}` + "\n\n")
		out, tokenUsage, _ := streamThrough(t, sse, 0, nil)
		r := accumulateAnthropicSSE(t, out)
		require.Equal(t, []string{
			"message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "error",
		}, r.events)
		requireAPIError(t, r, "upstream generation failed with finish reason MALFORMED_FUNCTION_CALL: Malformed function call: Read(")
		in, _ := tokenUsage.InputTokens()
		require.Equal(t, uint32(9), in)
	})

	t.Run("finish reason only chunk without content", func(t *testing.T) {
		sse := []byte(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]}}]}` + "\n\n" +
			`data: {"candidates":[{"finishReason":"MAX_TOKENS"}]}` + "\n\n")
		out, _, _ := streamThrough(t, sse, 0, nil)
		r := accumulateAnthropicSSE(t, out)
		require.Equal(t, anthropic.StopReasonMaxTokens, r.stopReason)
		require.Equal(t, []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "done"}}}, r.content)
	})

	t.Run("trailing event without a delimiter is handled at the end of the stream", func(t *testing.T) {
		sse := geminiSSE(t, "\n\n", []*genai.Part{{Text: "Hello "}}, []*genai.Part{{Text: "world"}})
		trimmed := bytes.TrimSuffix(sse, []byte("\n\n"))
		whole, _, _ := streamThrough(t, sse, 0, nil)
		for _, pieceSize := range []int{0, 7} {
			out, _, _ := streamThrough(t, trimmed, pieceSize, nil)
			require.Equal(t, string(whole), string(out), "piece size %d", pieceSize)
		}
		r := accumulateAnthropicSSE(t, whole)
		require.Equal(t, anthropic.StopReasonEndTurn, r.stopReason)
		require.Equal(t, []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "Hello world"}}}, r.content)
	})

	t.Run("multiple candidates use the first one", func(t *testing.T) {
		sse := []byte(`data: {"candidates":[` +
			`{"index":0,"content":{"role":"model","parts":[{"text":"first"}]},"finishReason":"STOP"},` +
			`{"index":1,"content":{"role":"model","parts":[{"text":"second"}]},"finishReason":"OTHER"}]}` + "\n\n")
		r := accumulateAnthropicSSE(t, func() []byte { out, _, _ := streamThrough(t, sse, 0, nil); return out }())
		require.Equal(t, anthropic.StopReasonEndTurn, r.stopReason)
		require.Equal(t, []anthropic.MessagesContentBlock{{Text: &anthropic.TextBlock{Type: "text", Text: "first"}}}, r.content)
	})

	t.Run("mid-stream error becomes an Anthropic error event", func(t *testing.T) {
		sse := []byte(`data: {"candidates":[{"content":{"role":"model","parts":[{"text":"partial"}]}}]}` + "\n\n")
		sse = append(sse, []byte(`data: {"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}`+"\n\n"+`data: {"candidates":[]}`+"\n\n")...)
		out, _, _ := streamThrough(t, sse, 0, nil)
		r := accumulateAnthropicSSE(t, out)
		require.Equal(t, []string{"message_start", "content_block_start", "content_block_delta", "error"}, r.events)
		var errResp anthropic.ErrorResponse
		require.NoError(t, json.Unmarshal(r.errorBody, &errResp))
		require.Equal(t, "error", errResp.Type)
		require.Equal(t, anthropic.ErrorResponseMessage{Type: "rate_limit_error", Message: "RESOURCE_EXHAUSTED: quota"}, errResp.Error)
	})

	t.Run("prompt blocked without a finish reason is a refusal", func(t *testing.T) {
		out, _, _ := streamThrough(t, []byte(`data: {"promptFeedback":{"blockReason":"SAFETY"}}`+"\r\n\r\n"), 0, nil)
		r := accumulateAnthropicSSE(t, out)
		require.Equal(t, anthropic.StopReasonRefusal, r.stopReason)
		require.Equal(t, []string{"message_start", "message_delta", "message_stop"}, r.events)
	})

	t.Run("malformed chunk", func(t *testing.T) {
		tr := newStreamingGCPVertexAITranslator(t)
		_, _, _, _, err := tr.ResponseBody(nil, strings.NewReader("data: {not json\n\n"), false, nil)
		require.ErrorContains(t, err, "failed to decode Gemini stream chunk")
	})

	t.Run("stream state not initialized", func(t *testing.T) {
		tr := &anthropicToGCPVertexAITranslator{stream: true}
		_, _, _, _, err := tr.ResponseBody(nil, strings.NewReader(""), true, nil)
		require.ErrorContains(t, err, "stream state not initialized")
	})
}

// TestGeminiThoughtSignatureRoundTrip verifies that a Gemini response translated to Anthropic blocks and
// echoed back by the client as the next assistant message restores every signature on the same part.
func TestGeminiThoughtSignatureRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name     string
		chunks   [][]*genai.Part
		expected []*genai.Part
	}{
		{
			name: "parallel function calls",
			chunks: [][]*genai.Part{
				{{Text: "plan", Thought: true}},
				{
					{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{"k": "v"}}, ThoughtSignature: []byte("s1")},
					{FunctionCall: &genai.FunctionCall{ID: "b", Name: "B", Args: map[string]any{}}},
				},
			},
			expected: []*genai.Part{
				{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{"k": "v"}}, ThoughtSignature: []byte("s1")},
				{FunctionCall: &genai.FunctionCall{ID: "b", Name: "B", Args: map[string]any{}}},
			},
		},
		{
			name: "text then signed function call then text",
			chunks: [][]*genai.Part{
				{{Text: "I will read it."}},
				{{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: []byte("s1")}},
				{{Text: "Then summarize."}},
			},
			expected: []*genai.Part{
				{Text: "I will read it."},
				{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}, ThoughtSignature: []byte("s1")},
				{Text: "Then summarize."},
			},
		},
		{
			name:     "text-only answer signed on the last text part carries no signature",
			chunks:   [][]*genai.Part{{{Text: "Hello "}}, {{Text: "world", ThoughtSignature: []byte("s1")}}},
			expected: []*genai.Part{{Text: "Hello world"}},
		},
		{
			name:     "text-only answer signed on an empty closing part carries no signature",
			chunks:   [][]*genai.Part{{{Text: "Hello "}}, {{Text: "world"}}, {{ThoughtSignature: []byte("s1")}}},
			expected: []*genai.Part{{Text: "Hello world"}},
		},
		{
			name:     "unsigned thought summary before the answer restores no signature",
			chunks:   [][]*genai.Part{{{Text: "hmm", Thought: true}}, {{Text: "answer"}}, {{ThoughtSignature: []byte("s1")}}},
			expected: []*genai.Part{{Text: "answer"}},
		},
		{
			name: "unsigned thought summary before an unsigned function call restores no signature",
			chunks: [][]*genai.Part{
				{{Text: "hmm", Thought: true}},
				{{Text: "Reading."}},
				{{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}}},
			},
			expected: []*genai.Part{
				{Text: "Reading."},
				{FunctionCall: &genai.FunctionCall{ID: "a", Name: "A", Args: map[string]any{}}},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var parts []*genai.Part
			for _, c := range tc.chunks {
				parts = append(parts, c...)
			}
			nonStreaming, _ := geminiPartsToAnthropicContent(parts)
			out, _, _ := streamThrough(t, geminiSSE(t, "\r\n\r\n", tc.chunks...), 0, nil)
			streaming := accumulateAnthropicSSE(t, out).content
			require.Equal(t, nonStreaming, streaming)

			// The client echoes the content blocks back as the assistant message.
			echoed, err := json.Marshal(streaming)
			require.NoError(t, err)
			var blocks []anthropic.ContentBlockParam
			require.NoError(t, json.Unmarshal(echoed, &blocks))
			restored := anthropicAssistantContentToGeminiParts(&anthropic.MessageContent{Array: blocks})
			require.Equal(t, tc.expected, restored)
		})
	}
}

func TestAnthropicToGCPVertexAI_ResponseError(t *testing.T) {
	for _, tc := range []struct {
		name            string
		status          string
		body            string
		expectedType    string
		expectedMessage string
	}{
		{
			name:            "invalid argument with details",
			status:          "400",
			body:            `{"error":{"code":400,"message":"bad signature","status":"INVALID_ARGUMENT","details":[{"reason":"x"}]}}`,
			expectedType:    "invalid_request_error",
			expectedMessage: "INVALID_ARGUMENT: bad signature\nDetails: [{\"reason\":\"x\"}]",
		},
		{
			name:            "streaming endpoints return an array",
			status:          "429",
			body:            `[{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}]`,
			expectedType:    "rate_limit_error",
			expectedMessage: "RESOURCE_EXHAUSTED: quota",
		},
		{
			name:            "unavailable maps to overloaded",
			status:          "503",
			body:            `{"error":{"code":503,"message":"overloaded","status":"UNAVAILABLE"}}`,
			expectedType:    "overloaded_error",
			expectedMessage: "UNAVAILABLE: overloaded",
		},
		{
			name:            "status taken from the body when the header is missing",
			body:            `{"error":{"code":404,"message":"no model","status":"NOT_FOUND"}}`,
			expectedType:    "not_found_error",
			expectedMessage: "NOT_FOUND: no model",
		},
		{name: "plain text body", status: "500", body: "upstream exploded", expectedType: "api_error", expectedMessage: "upstream exploded"},
		{name: "authentication", status: "401", body: "no", expectedType: "authentication_error", expectedMessage: "no"},
		{name: "permission", status: "403", body: "no", expectedType: "permission_error", expectedMessage: "no"},
		{name: "too large", status: "413", body: "big", expectedType: "request_too_large", expectedMessage: "big"},
		{name: "timeout", status: "504", body: "slow", expectedType: "timeout_error", expectedMessage: "slow"},
		{name: "other client error", status: "409", body: "conflict", expectedType: "invalid_request_error", expectedMessage: "conflict"},
		{name: "empty array", status: "502", body: "[]", expectedType: "api_error", expectedMessage: "[]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := NewAnthropicToGCPVertexAITranslator("")
			headers, body, err := tr.ResponseError(map[string]string{statusHeaderName: tc.status}, strings.NewReader(tc.body))
			require.NoError(t, err)
			require.Contains(t, headers, internalapi.Header{contentTypeHeaderName, "application/json"})
			require.Equal(t, strconv.Itoa(len(body)), headerValue(headers, contentLengthHeaderName))
			var errResp anthropic.ErrorResponse
			require.NoError(t, json.Unmarshal(body, &errResp))
			require.Equal(t, "error", errResp.Type)
			require.Equal(t, anthropic.ErrorResponseMessage{Type: tc.expectedType, Message: tc.expectedMessage}, errResp.Error)
		})
	}
}

func TestAnthropicToGCPVertexAI_RedactAnthropicBody(t *testing.T) {
	tr := NewAnthropicToGCPVertexAITranslator("").(AnthropicResponseRedactor)
	require.Nil(t, tr.RedactAnthropicBody(nil))

	resp := &anthropic.MessagesResponse{ID: "msg_1", Content: []anthropic.MessagesContentBlock{
		{Text: &anthropic.TextBlock{Type: "text", Text: "secret"}},
	}}
	redacted := tr.RedactAnthropicBody(resp)
	require.Equal(t, "msg_1", redacted.ID)
	require.NotEqual(t, "secret", redacted.Content[0].Text.Text)
	require.Equal(t, "secret", resp.Content[0].Text.Text)
}
