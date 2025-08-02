// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"

	client "github.com/ashishdatta/workers-ai-golang/workers-ai"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/pkg/errors"
)

const provider = "workersai"

// WorkersAI holds the shared client instance.
type WorkersAI struct {
	client  *client.Client
	mu      sync.Mutex
	initted bool
}

// generator is the internal struct that implements the model generation logic.
type generator struct {
	model  string
	client *client.Client
}

// Name returns the name of the plugin.
func (w *WorkersAI) Name() string {
	return provider
}

// Init initializes the Workers AI plugin and the shared client.
func (w *WorkersAI) Init(ctx context.Context, g *genkit.Genkit) (err error) {
	if w == nil {
		w = &WorkersAI{}
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.initted {
		return errors.New("workersai plugin already initialized")
	}

	defer func() {
		if err != nil {
			err = fmt.Errorf("WorkersAI.Init: %w", err)
		}
	}()

	apiToken := os.Getenv("CLOUDFLARE_API_TOKEN")
	if apiToken == "" {
		return errors.New("Workers AI requires setting CLOUDFLARE_API_TOKEN in the environment")
	}

	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if accountID == "" {
		return errors.New("Workers AI requires setting CLOUDFLARE_ACCOUNT_ID in the environment")
	}

	// Initialize the client from your library.
	w.client = client.NewClient(accountID, apiToken)
	w.initted = true

	// You can set debug mode for the client if needed.
	if os.Getenv("GENKIT_ENV") == "dev" {
		w.client.SetDebug(true)
	}

	// Register known models here.
	for name, info := range supportedWorkersAIModels {
		w.defineModel(g, name, info)
	}

	return nil
}

// defineModel is a helper to register a model with Genkit.
func (w *WorkersAI) defineModel(g *genkit.Genkit, name string, info ai.ModelInfo) {
	gen := &generator{
		model:  name,
		client: w.client,
	}
	genkit.DefineModel(g, provider, name, &info, gen.generate)
}

// DefineModel defines a Workers AI model for use in Genkit.
func (w *WorkersAI) DefineModel(g *genkit.Genkit, name string, info *ai.ModelInfo) {
	if !w.initted {
		panic("Workers AI plugin not initialized")
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	var mi ai.ModelInfo
	if info != nil {
		mi = *info
	} else {
		mi = ai.ModelInfo{
			Label: "Workers AI - " + name,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				SystemRole: true,
				Media:      false,
				Tools:      true,
			},
		}
	}
	w.defineModel(g, name, mi)
}

// generate is the core translation layer between Genkit and the Workers AI client.
func (gen *generator) generate(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// 1. Convert Genkit Tools to the client library's Tool format.
	clientTools, err := toClientTools(input.Tools)
	if err != nil {
		return nil, errors.Wrap(err, "failed to convert tools")
	}

	// 2. Convert Genkit Messages to the client library's Message format.
	clientMessages, err := toClientMessages(input.Messages)
	if err != nil {
		return nil, errors.Wrap(err, "failed to convert messages")
	}

	// 3. Call the client library. All HTTP and response format complexity is handled here.
	resp, err := gen.client.ChatWithTools(gen.model, clientMessages, clientTools)
	if err != nil {
		return nil, errors.Wrap(err, "workersai client failed")
	}

	if !resp.Success {
		return nil, fmt.Errorf("workersai API returned an error: %v", resp.Errors)
	}

	// 4. Process the response.
	modelResponse := &ai.ModelResponse{
		Request: input,
		Usage:   &ai.GenerationUsage{}, // Usage will be populated below.
	}

	// Populate usage data regardless of response format.
	if resp.IsLegacyResult {
		modelResponse.Usage.InputTokens = resp.LegacyResponse.Usage.PromptTokens
		modelResponse.Usage.OutputTokens = resp.LegacyResponse.Usage.CompletionTokens
	} else {
		modelResponse.Usage.InputTokens = resp.ChatCompletionResponse.Usage.PromptTokens
		modelResponse.Usage.OutputTokens = resp.ChatCompletionResponse.Usage.CompletionTokens
	}

	// Check if the response contains tool calls.
	toolCalls := resp.GetToolCalls()
	if len(toolCalls) > 0 {
		toolRequestParts, err := toGenkitToolRequestParts(toolCalls)
		if err != nil {
			return nil, err
		}

		modelResponse.Message = &ai.Message{Role: ai.RoleModel, Content: toolRequestParts}
		modelResponse.FinishReason = ai.FinishReasonStop
	} else {
		// Handle a standard text response.
		modelResponse.Message = &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart(resp.GetContent())},
		}
		modelResponse.FinishReason = ai.FinishReasonStop
	}

	return modelResponse, nil
}

// simplifyArguments adapts verbose model arguments into the simple format Genkit expects.
func simplifyArguments(argsJSON string) (map[string]any, error) {
	var rawArgs map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &rawArgs); err != nil {
		return nil, fmt.Errorf("failed to unmarshal tool arguments from string: %w", err)
	}

	simplifiedArgs := make(map[string]any)
	for key, val := range rawArgs {
		if argObject, ok := val.(map[string]any); ok {
			if value, hasValue := argObject["value"]; hasValue {
				simplifiedArgs[key] = value
				continue
			}
		}
		simplifiedArgs[key] = val
	}
	return simplifiedArgs, nil
}

// toGenkitToolRequestParts adapts the tool calls from the client library's response
// into a slice of *ai.Part suitable for Genkit. It handles both simple and verbose
// argument formats from different models.
func toGenkitToolRequestParts(calls []client.ToolCall) ([]*ai.Part, error) {
	var toolRequestParts []*ai.Part
	for _, call := range calls {
		// First, unmarshal the arguments string from the model into a raw map.
		var rawArgs map[string]any
		if err := json.Unmarshal([]byte(call.Function.Arguments), &rawArgs); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tool arguments for '%s': %w", call.Function.Name, err)
		}

		// create a new map with simplified arguments.
		simplifiedArgs := make(map[string]any)
		for key, val := range rawArgs {
			// For each argument, check if it's a map containing a "value" key.
			// This is the signature of the verbose format.
			if argObject, ok := val.(map[string]any); ok {
				if value, hasValue := argObject["value"]; hasValue {
					// If it is, use the inner value.
					simplifiedArgs[key] = value
					continue
				}
			}
			// Otherwise, use the value as-is (for the simple format).
			simplifiedArgs[key] = val
		}

		// Create the ToolRequest struct that Genkit expects.
		tr := &ai.ToolRequest{
			Ref:   call.ID,
			Name:  call.Function.Name,
			Input: simplifiedArgs,
		}

		toolRequestParts = append(toolRequestParts, ai.NewToolRequestPart(tr))
	}
	return toolRequestParts, nil
}

// toClientTools converts Genkit tool definitions to the client library's format.
func toClientTools(defs []*ai.ToolDefinition) ([]client.Tool, error) {
	if len(defs) == 0 {
		return nil, nil
	}
	var tools []client.Tool
	for _, def := range defs {
		var params client.FunctionParameters
		if def.InputSchema != nil {
			schemaBytes, err := json.Marshal(def.InputSchema)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to marshal schema for tool %s", def.Name)
			}
			if err := json.Unmarshal(schemaBytes, &params); err != nil {
				return nil, errors.Wrapf(err, "failed to unmarshal schema for tool %s", def.Name)
			}
		}

		tools = append(tools, client.Tool{
			Type: "function",
			Function: client.FunctionDefinition{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  params,
			},
		})
	}
	return tools, nil
}
func toClientMessages(messages []*ai.Message) ([]client.Message, error) {
	var clientMsgs []client.Message
	for _, msg := range messages {
		switch msg.Role {
		case ai.RoleTool:
			// Handle the tool's response.
			for _, part := range msg.Content {
				if part.IsToolResponse() {
					outputBytes, err := json.Marshal(part.ToolResponse.Output)
					if err != nil {
						return nil, errors.Wrapf(err, "failed to marshal tool output for %s", part.ToolResponse.Name)
					}

					clientMsgs = append(clientMsgs, client.ToolMessage{
						Role:       "tool",
						Content:    string(outputBytes),
						ToolCallID: part.ToolResponse.Ref, // Read the ID back from Ref
					})
				}
			}
		case ai.RoleModel:
			// Handle the assistant's previous message (the tool request).
			var toolCalls []client.ToolCall
			for _, part := range msg.Content {
				if part.IsToolRequest() {
					// We must convert Genkit's request back to the client library's format.
					// This is crucial for maintaining conversation history.
					argsBytes, err := json.Marshal(part.ToolRequest.Input)
					if err != nil {
						return nil, errors.Wrapf(err, "failed to marshal tool input for %s", part.ToolRequest.Name)
					}
					toolCalls = append(toolCalls, client.ToolCall{
						ID:   part.ToolRequest.Ref, // Pass the ID along
						Type: "function",
						Function: client.FunctionToCall{
							Name:      part.ToolRequest.Name,
							Arguments: string(argsBytes),
						},
					})
				}
			}
			// Add the assistant message with its tool calls to the history.
			if len(toolCalls) > 0 {
				clientMsgs = append(clientMsgs, client.ResponseMessage{
					Role:      "assistant",
					ToolCalls: toolCalls,
					Content:   new(string),
				})
			} else if msg.Text() != "" {
				clientMsgs = append(clientMsgs, client.ChatMessage{
					Role:    "assistant",
					Content: msg.Text(),
				})
			}

		case ai.RoleUser, ai.RoleSystem:
			// Handle standard user or system messages.
			clientMsgs = append(clientMsgs, client.ChatMessage{
				Role:    convertRole(msg.Role),
				Content: msg.Text(),
			})
		}
	}
	return clientMsgs, nil
}

// convertRole converts Genkit roles to the client library's format.
func convertRole(role ai.Role) string {
	switch role {
	case ai.RoleUser:
		return "user"
	case ai.RoleModel:
		return "assistant"
	case ai.RoleSystem:
		return "system"
	case ai.RoleTool:
		return "tool"
	default:
		return "user"
	}
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
	return ai.NewModelRef("blarg"+provider+"/"+name, nil)
}
