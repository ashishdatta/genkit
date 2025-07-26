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
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"

	workersai "github.com/ashishdatta/workers-ai-golang/workers-ai"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/kortschak/utter"
	"github.com/pkg/errors"
)

const provider = "workersai"

// WorkersAI holds the shared client instance.
type WorkersAI struct {
	client  *workersai.Client
	mu      sync.Mutex
	initted bool
}

type generator struct {
	model  string
	client *workersai.Client
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
	w.client = workersai.NewClient(accountID, apiToken)
	w.initted = true
	w.client.Debug = true

	// You can still register known models here if you have a predefined list.
	for name, info := range supportedWorkersAIModels {
		w.defineModel(g, name, info)
	}

	return nil
}

func (w *WorkersAI) defineModel(g *genkit.Genkit, name string, info ai.ModelInfo) ai.Model {
	gen := &generator{
		model:  name,
		client: w.client,
	}

	return genkit.DefineModel(g, provider, name, &info, gen.generate)
}

// DefineModel defines a Workers AI model.
func (w *WorkersAI) DefineModel(g *genkit.Genkit, name string, info *ai.ModelInfo) ai.Model {
	if w.client == nil {
		panic("Workers AI plugin not initialized")
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	var mi ai.ModelInfo
	if info != nil {
		mi = *info
	} else {
		// Default model info, assuming tool support.
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

	return w.defineModel(g, name, mi)
}

// generate is now the core translation layer.
func (gen *generator) generate(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	// 1. Convert Genkit Tools to the client library's FunctionTool format.
	clientTools, err := toClientTools(input.Tools)
	if err != nil {
		return nil, errors.Wrap(err, "failed to convert tools")
	}

	// 2. Convert Genkit Messages to the client library's Message format.
	clientMessages, err := toClientMessages(input.Messages)
	if err != nil {
		return nil, errors.Wrap(err, "failed to convert messages")
	}

	// 3. Call the client library. All HTTP complexity is handled here.
	resp, err := gen.client.ChatWithTools(gen.model, clientMessages, clientTools)
	if err != nil {
		return nil, errors.Wrap(err, "workersai client failed")
	}

	if !resp.Success {
		return nil, fmt.Errorf("workersai API returned an error: %v", resp.Errors)
	}

	var promptTokens, completionTokens int
	if resp.IsLegacyResult {
		promptTokens = resp.LegacyResponse.PromptTokens
		completionTokens = resp.LegacyResponse.CompletionTokens
	} else {
		promptTokens = resp.OpenAIResponse.Usage.PromptTokens
		completionTokens = resp.OpenAIResponse.Usage.CompletionTokens
	}

	// 4. Process the response.
	modelResponse := &ai.ModelResponse{
		Request: input,
		Usage: &ai.GenerationUsage{
			InputTokens:  promptTokens,
			OutputTokens: completionTokens,
		},
	}

	// Check if the response contains tool calls.
	toolCalls := resp.GetToolCalls()

	utter.Dump(toolCalls)
	if len(toolCalls) > 0 {
		var toolRequestParts []*ai.Part
		for _, call := range toolCalls {
			// The client library's `Arguments` field is a string, which we
			// wrap in json.RawMessage for Genkit.
			toolRequest := &ai.ToolRequest{
				Name:  call.Function.Name,
				Input: call.Function.Arguments,
			}

			toolRequestParts = append(toolRequestParts, ai.NewToolRequestPart(toolRequest))
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

func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, provider, name) != nil
}

func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, provider, name)
}

func ModelRef(name string) ai.ModelRef {
	return ai.NewModelRef(provider+"/"+name, nil)
}

// toClientTools converts Genkit tool definitions to the client library's format.
func toClientTools(defs []*ai.ToolDefinition) ([]workersai.FunctionTool, error) {
	if len(defs) == 0 {
		return nil, nil
	}
	var tools []workersai.FunctionTool
	for _, def := range defs {
		// This struct is used to easily parse the schema from Genkit's map[string]any.
		type schemaDef struct {
			Type       string                          `json:"type"`
			Properties map[string]*workersai.Parameter `json:"properties"`
			Required   []string                        `json:"required"`
		}

		var schema schemaDef
		if def.InputSchema != nil {
			schemaBytes, err := json.Marshal(def.InputSchema)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to marshal schema for tool %s", def.Name)
			}
			if err := json.Unmarshal(schemaBytes, &schema); err != nil {
				return nil, errors.Wrapf(err, "failed to unmarshal schema for tool %s", def.Name)
			}
		}

		tools = append(tools, workersai.FunctionTool{
			Type: "function",
			Function: struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Parameters  struct {
					Type       string                          `json:"type"`
					Required   []string                        `json:"required"`
					Properties map[string]*workersai.Parameter `json:"properties"`
				} `json:"parameters"`
			}{
				Name:        def.Name,
				Description: def.Description,
				Parameters: struct {
					Type       string                          `json:"type"`
					Required   []string                        `json:"required"`
					Properties map[string]*workersai.Parameter `json:"properties"`
				}{
					Type:       schema.Type,
					Required:   schema.Required,
					Properties: schema.Properties,
				},
			},
		})
	}
	return tools, nil
}

// toClientMessages converts Genkit messages to the client library's format.
func toClientMessages(messages []*ai.Message) ([]workersai.Message, error) {
	var clientMsgs []workersai.Message
	for _, msg := range messages {
		text := msg.Text()

		// Handle tool response messages.
		if msg.Role == ai.RoleTool {
			clientMsgs = append(clientMsgs, workersai.Message{
				Role:    "tool",
				Content: text,
				// Assuming the ToolResponse Name corresponds to a ToolCallID.
				// The client library schema expects a ToolCallID here.
				// This might need adjustment if the API requires a specific ID from the request.
				// ToolCallID: part.ToolResponse.Name,
			})
			continue
		}

		clientMsgs = append(clientMsgs, workersai.Message{
			Role:    convertRole(msg.Role),
			Content: text,
		})
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
