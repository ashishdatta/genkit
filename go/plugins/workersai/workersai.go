// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package workersai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/invopop/jsonschema"
	"github.com/pkg/errors"
)

const provider = "workersai"

// WorkersAI provides configuration options for the Workers AI plugin.
type WorkersAI struct {
	APIToken  string // API token for Cloudflare Workers AI. If empty, the value of the environment variable CLOUDFLARE_API_TOKEN will be used.
	AccountID string // Cloudflare account ID. If empty, the value of the environment variable CLOUDFLARE_ACCOUNT_ID will be used.
	BaseURL   string // Base URL for the API. If empty, defaults to "https://api.cloudflare.com/client/v4".

	httpClient *http.Client
	mu         sync.Mutex
	initted    bool
}

// workersAIRequest represents the request structure for Workers AI API.
type workersAIRequest struct {
	Messages []workersAIMessage `json:"messages,omitempty"`
	Prompt   string             `json:"prompt,omitempty"`
	Stream   bool               `json:"stream,omitempty"`
	Tools    []workersAITool    `json:"tools,omitempty"` // NEW: For sending tool definitions
	// https://developers.cloudflare.com/workers-ai/features/function-calling/embedded/examples/fetch/
}

type workersAIToolCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type workersAITool struct {
	Type     string            `json:"type"`
	Function workersAIFunction `json:"function"`
}

type workersAIFunction struct {
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	Parameters  *jsonschema.Schema `json:"parameters"`
}

// workersAIMessage represents a message in the Workers AI API format.
type workersAIMessage struct {
	Role      string              `json:"role"`
	Content   string              `json:"content"`
	ToolCalls []workersAIToolCall `json:"tool_calls,omitempty"` // NEW: For assistant messages requesting tool calls
}

// workersAIResponse represents the response structure from Workers AI API.
type workersAIResponse struct {
	Success bool            `json:"success"`
	Result  workersAIResult `json:"result"`
	Errors  []struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
	Messages []interface{} `json:"messages"`
}

type workersAIResult struct {
	Response  string              `json:"response"`
	ToolCalls []workersAIToolCall `json:"tool_calls,omitempty"`
}

// Result represents the result structure in the Workers AI response.
type Result struct {
	Response string `json:"response"`
}

// generator holds the configuration for generating responses.
type generator struct {
	model     string
	apiToken  string
	accountID string
	baseURL   string
	client    *http.Client
}

// Name returns the name of the plugin.
func (w *WorkersAI) Name() string {
	return provider
}

// Init initializes the Workers AI plugin.
func (w *WorkersAI) Init(ctx context.Context, g *genkit.Genkit) error {
	if w == nil {
		w = &WorkersAI{}
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.initted {
		return errors.New("Workers AI plugin already initialized")
	}

	// Set API token from environment if not provided
	apiToken := w.APIToken
	if apiToken == "" {
		apiToken = os.Getenv("CLOUDFLARE_API_TOKEN")
		if apiToken == "" {
			return errors.New("Workers AI requires setting CLOUDFLARE_API_TOKEN in the environment or providing APIToken in config")
		}
	}

	// Set account ID from environment if not provided
	accountID := w.AccountID
	if accountID == "" {
		accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID")
		if accountID == "" {
			return errors.New("Workers AI requires setting CLOUDFLARE_ACCOUNT_ID in the environment or providing AccountID in config")
		}
	}

	// Set base URL if not provided
	baseURL := w.BaseURL
	if baseURL == "" {
		baseURL = "https://api.cloudflare.com/client/v4"
	}

	// Create HTTP client if not provided
	if w.httpClient == nil {
		w.httpClient = &http.Client{
			Timeout: 30 * time.Second,
		}
	}

	w.APIToken = apiToken
	w.AccountID = accountID
	w.BaseURL = baseURL
	w.initted = true

	// Register known models
	for modelName, modelInfo := range supportedWorkersAIModels {
		_ = w.defineModel(g, modelName, modelInfo)
	}

	return nil
}

// DefineModel defines a Workers AI model with the given name and configuration.
func (w *WorkersAI) DefineModel(g *genkit.Genkit, name string, info *ai.ModelInfo) ai.Model {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.initted {
		panic("Workers AI plugin not initialized")
	}

	var mi ai.ModelInfo
	if info != nil {
		mi = *info
	} else {
		mi = ai.ModelInfo{
			Label: "Workers AI - " + name,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				SystemRole: true,
				Media:      false, // Most Workers AI models don't support media yet
				Tools:      false, // Tool calling support varies by model
			},
			Versions: []string{},
		}
	}

	return w.defineModel(g, name, mi)
}

func (w *WorkersAI) defineModel(g *genkit.Genkit, name string, info ai.ModelInfo) ai.Model {
	gen := &generator{
		model:     name,
		apiToken:  w.APIToken,
		accountID: w.AccountID,
		baseURL:   w.BaseURL,
		client:    w.httpClient,
	}

	return genkit.DefineModel(g, provider, name, &info, gen.generate)
}

// IsDefinedModel reports whether a model is defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, provider, name) != nil
}

// Model returns the [ai.Model] with the given name.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, provider, name)
}

// ModelRef creates a new ModelRef for a Workers AI model.
func ModelRef(name string) ai.ModelRef {
	return ai.NewModelRef(provider+"/"+name, nil)
}

// generate performs the actual generation using the Workers AI API.
func (gen *generator) generate(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// Build the request
	req := &workersAIRequest{
		Stream: cb != nil,
	}

	if len(input.Tools) > 0 {
		var err error
		req.Tools, err = gen.toWorkersAITools(input.Tools)
		if err != nil {
			return nil, err
		}
	}

	// Convert messages to Workers AI format
	messages, err := gen.toWorkersAIMessages(input.Messages)
	if err != nil {
		return nil, err
	}

	req.Messages = messages

	// Make the API request
	url := fmt.Sprintf("%s/accounts/%s/ai/run/@cf/%s", gen.baseURL, gen.accountID, gen.model)
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+gen.apiToken)

	resp, err := gen.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Handle streaming response
	if cb != nil {
		return gen.handleStreamingResponse(ctx, resp, cb, input)
	}

	// Handle non-streaming response
	return gen.handleResponse(resp, input)
}

// handleResponse processes a non-streaming response from Workers AI.
func (gen *generator) handleResponse(resp *http.Response, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var workerResp workersAIResponse
	if err := json.Unmarshal(body, &workerResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if !workerResp.Success {
		errMsg := "API request failed"
		if len(workerResp.Errors) > 0 {
			errMsg = workerResp.Errors[0].Message
		}
		return nil, errors.New(errMsg)
	}

	// if its a tool call return
	if len(workerResp.Result.ToolCalls) > 0 {
		var toolRequestParts []*ai.Part
		for _, call := range workerResp.Result.ToolCalls {
			tr := &ai.ToolRequest{
				Name:  call.Name,
				Input: call.Arguments,
			}
			toolRequestParts = append(toolRequestParts, ai.NewToolRequestPart(tr))
		}

		response := &ai.ModelResponse{
			Request:      input,
			FinishReason: ai.FinishReasonStop,
			Message: &ai.Message{
				Role:    ai.RoleModel,
				Content: toolRequestParts,
			},
		}

		return response, nil
	}

	response := &ai.ModelResponse{
		Request:      input,
		FinishReason: ai.FinishReason("stop"),
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(workerResp.Result.Response)},
		},
		Usage: &ai.GenerationUsage{}, // Workers AI doesn't provide usage metrics in the response
	}

	return response, nil
}

// handleStreamingResponse processes a streaming response from Workers AI.
func (gen *generator) handleStreamingResponse(ctx context.Context, resp *http.Response, cb func(context.Context, *ai.ModelResponseChunk) error, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	// Note: Workers AI streaming implementation would depend on their specific streaming format
	// For now, we'll fall back to non-streaming behavior
	return gen.handleResponse(resp, input)
}

func (gen *generator) toWorkersAIMessages(messages []*ai.Message) ([]workersAIMessage, error) {
	var wmsgs []workersAIMessage

	for _, msg := range messages {
		var text strings.Builder
		var toolCalls []workersAIToolCall

		for _, part := range msg.Content {
			switch {
			case part.IsText():
				text.WriteString(part.Text)
			case part.IsToolRequest():
				inputRawMsg, err := asJSONRawMessage(part.ToolRequest.Input)
				if err != nil {
					return nil, errors.Wrap(err, "error marshalling tool request input")
				}

				toolCalls = append(toolCalls, workersAIToolCall{
					Name:      part.ToolRequest.Name,
					Arguments: inputRawMsg,
				})
			case part.IsToolResponse():
				outputRawMsg, err := asJSONRawMessage(part.ToolResponse.Output)
				if err != nil {
					return nil, errors.Wrap(err, "error marshalling tool response output")
				}

				wmsgs = append(wmsgs, workersAIMessage{
					Role:    "tool",
					Content: string(outputRawMsg),
				})
			}
		}

		if text.Len() > 0 || len(toolCalls) > 0 {
			wmsgs = append(wmsgs, workersAIMessage{
				Role:      convertRole(msg.Role),
				Content:   text.String(),
				ToolCalls: toolCalls,
			})
		}
	}

	return wmsgs, nil
}

// ListActions returns the list of available actions for this plugin.
func (w *WorkersAI) ListActions(ctx context.Context) []core.ActionDesc {
	var actions []core.ActionDesc

	for modelName, modelInfo := range supportedWorkersAIModels {
		metadata := map[string]any{
			"model": map[string]any{
				"supports": map[string]any{
					"media":       modelInfo.Supports.Media,
					"multiturn":   modelInfo.Supports.Multiturn,
					"systemRole":  modelInfo.Supports.SystemRole,
					"tools":       modelInfo.Supports.Tools,
					"toolChoice":  false, // Workers AI doesn't support tool choice
					"constrained": false, // Workers AI doesn't support constrained generation
				},
				"versions": modelInfo.Versions,
				"stage":    string(modelInfo.Stage),
			},
		}
		metadata["label"] = modelInfo.Label

		actions = append(actions, core.ActionDesc{
			Type:     core.ActionTypeModel,
			Name:     fmt.Sprintf("%s/%s", provider, modelName),
			Key:      fmt.Sprintf("/%s/%s/%s", core.ActionTypeModel, provider, modelName),
			Metadata: metadata,
		})
	}

	return actions
}

// ResolveAction resolves an action by type and name.
func (w *WorkersAI) ResolveAction(g *genkit.Genkit, atype core.ActionType, name string) error {
	switch atype {
	case core.ActionTypeModel:
		models := supportedWorkersAIModels
		if modelInfo, exists := models[name]; exists {
			w.DefineModel(g, name, &modelInfo)
		} else {
			// Define with default info if model not in known list
			w.DefineModel(g, name, nil)
		}
	}
	return nil
}

func (gen *generator) toWorkersAITools(input []*ai.ToolDefinition) ([]workersAITool, error) {
	var tools []workersAITool
	for _, tool := range input {
		var schema *jsonschema.Schema
		if tool.InputSchema != nil {
			// Marshal the map to JSON bytes.
			schemaBytes, err := json.Marshal(tool.InputSchema)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal tool schema for %s: %w", tool.Name, err)
			}
			// Unmarshal the JSON bytes into the jsonschema.Schema struct.
			schema = new(jsonschema.Schema)
			if err := json.Unmarshal(schemaBytes, schema); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool schema for %s: %w", tool.Name, err)
			}
		}

		tools = append(tools, workersAITool{
			Type: "function",
			Function: workersAIFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  schema,
			},
		})
	}

	return tools, nil
}

// Helper functions

// convertRole converts Genkit roles to Workers AI roles.
func convertRole(role ai.Role) string {
	switch role {
	case ai.RoleUser:
		return "user"
	case ai.RoleModel:
		return "assistant"
	case ai.RoleSystem:
		return "system"
	default:
		return "user"
	}
}

func asJSONRawMessage(input any) (json.RawMessage, error) {
	var args json.RawMessage

	switch v := input.(type) {
	case json.RawMessage:
		args = v
	case []byte:
		args = v
	case string:
		args = []byte(v)
	default:
		// Fallback for other types: marshal them to JSON.
		return json.Marshal(v)
	}

	return args, nil
}
