package workersai

import (
	"context"
	"log"
	"math"
	"os"
	"strings"
	"testing"

	client "github.com/ashishdatta/workers-ai-golang/workers-ai"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func requireEnv(key string) (string, bool) {
	value, ok := os.LookupEnv(key)
	if !ok || value == "" {
		return "", false
	}

	return value, true
}

func TestWorkersAILive(t *testing.T) {
	ctx := context.Background()

	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&WorkersAI{}),
		// TODO: doesn't work with the mistralai model
		//genkit.WithDefaultModel("workersai/@cf/mistralai/mistral-small-3.1-24b-instruct"),
		//genkit.WithDefaultModel("workersai/@cf/meta/llama-4-scout-17b-16e-instruct"),
		genkit.WithDefaultModel("workersai/@cf/qwen/qwen3-30b-a3b-fp8"),
	)
	if err != nil {
		log.Fatal(err)
	}

	t.Run("generate", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Which country was Napoleon the emperor of? Name the country, nothing else"),
		)
		if err != nil {
			t.Fatal(err)
		}

		out := strings.ReplaceAll(resp.Message.Content[0].Text, "\n", "")
		const want = "France"
		if out != want {
			t.Errorf("got %q, expecting %q", out, want)
		}
		if resp.Request == nil {
			t.Error("Request field not set properly")
		}
	})

	// TODO: figure out why this isn't functional

	gablorkenTool := genkit.DefineTool(g, "gablorken", "use this tool when the user asks to calculate a gablorken, carefuly inspect the user input to determine which value from the prompt corresponds to the input structure",
		func(ctx *ai.ToolContext, input struct {
			Value int
			Over  float64
		},
		) (float64, error) {
			return math.Pow(float64(input.Value), input.Over), nil
		},
	)

	t.Run("tool", func(t *testing.T) {

		//tools := genkit.ListTools(g)
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what is a gablorken of 2 over 3.5? use the gablorken tool"),
			ai.WithTools(gablorkenTool),
			ai.WithMaxTurns(1),
			//ai.WithReturnToolRequests(true),
		)
		if err != nil {
			t.Fatal(err)
		}

		out := resp.Message.Content[0].Text
		const want = "11.31"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
	})

}

func TestToGenkitToolRequestParts(t *testing.T) {
	// Define test cases in a table for clarity and maintainability.
	testCases := []struct {
		name          string
		inputCalls    []client.ToolCall
		expectedParts []*ai.Part
		expectError   bool
		errorContains string
	}{
		{
			name: "should correctly parse simple argument format",
			inputCalls: []client.ToolCall{
				{
					ID:   "call_simple_123",
					Type: "function",
					Function: client.FunctionToCall{
						Name:      "get_weather",
						Arguments: `{"location": "Eindhoven, NL", "unit": "celsius"}`,
					},
				},
			},
			expectedParts: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{
					Name: "get_weather",
					Input: map[string]any{
						"location": "Eindhoven, NL",
						"unit":     "celsius",
					},
					Ref: "call_simple_123",
				}),
			},
			expectError: false,
		},
		{
			name: "should correctly parse and simplify verbose argument format",
			inputCalls: []client.ToolCall{
				{
					ID:   "call_verbose_456",
					Type: "function",
					Function: client.FunctionToCall{
						Name:      "gablorken",
						Arguments: `{"Over": {"type": "number", "value": 3.5}, "Value": {"type": "integer", "value": 2}}`,
					},
				},
			},
			expectedParts: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{
					Name: "gablorken",
					Input: map[string]any{
						"Over":  3.5,
						"Value": float64(2), // JSON numbers are unmarshaled as float64.
					},
					Ref: "call_verbose_456",
				}),
			},
			expectError: false,
		},
		{
			name: "should handle a mix of simple and verbose formats in a single call",
			inputCalls: []client.ToolCall{
				{
					Function: client.FunctionToCall{
						Name:      "get_weather_mixed",
						Arguments: `{"location": "Eindhoven, NL", "unit": {"type": "string", "value": "celsius"}}`,
					},
				},
			},
			expectedParts: []*ai.Part{
				ai.NewToolRequestPart(&ai.ToolRequest{
					Name: "get_weather_mixed",
					Input: map[string]any{
						"location": "Eindhoven, NL",
						"unit":     "celsius",
					},
				}),
			},
			expectError: false,
		},
		{
			name: "should return an error for malformed arguments JSON",
			inputCalls: []client.ToolCall{
				{
					Function: client.FunctionToCall{
						Name:      "get_weather",
						Arguments: `{"location": "Eindhoven, NL",`, // Missing closing brace
					},
				},
			},
			expectedParts: nil,
			expectError:   true,
			errorContains: "failed to unmarshal tool arguments",
		},
		{
			name:          "should return an empty slice for no tool calls",
			inputCalls:    []client.ToolCall{},
			expectedParts: []*ai.Part{},
			expectError:   false,
		},
	}

	// Iterate over the test cases and run them as sub-tests.
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Act: Call the function under test.
			parts, err := toGenkitToolRequestParts(tc.inputCalls)

			// Assert: Check the results.
			if tc.expectError {
				require.Error(t, err)
				if tc.errorContains != "" {
					assert.Contains(t, err.Error(), tc.errorContains)
				}
			} else {
				require.NoError(t, err)
				assert.Equal(t, len(tc.expectedParts), len(parts))
				if len(tc.expectedParts) > 0 {
					assert.Equal(t, tc.expectedParts[0].ToolRequest, parts[0].ToolRequest)
				}
			}
		})
	}
}

func TestToClientMessages(t *testing.T) {
	// Define the structure for our test cases
	testCases := []struct {
		name      string
		input     []*ai.Message
		expected  []interface{}
		expectErr bool
	}{
		{
			name: "Simple user message",
			input: []*ai.Message{
				ai.NewUserMessage(ai.NewTextPart("Hello, world!")),
			},
			expected: []interface{}{
				client.ChatMessage{Role: "user", Content: "Hello, world!"},
			},
			expectErr: false,
		},
		{
			name: "User and system messages",
			input: []*ai.Message{
				ai.NewSystemMessage(ai.NewTextPart("You are a helpful assistant.")),
				ai.NewUserMessage(ai.NewTextPart("What is Go?")),
			},
			expected: []interface{}{
				client.ChatMessage{Role: "system", Content: "You are a helpful assistant."},
				client.ChatMessage{Role: "user", Content: "What is Go?"},
			},
			expectErr: false,
		},
		{
			name: "Simple model (assistant) response",
			input: []*ai.Message{
				ai.NewModelMessage(ai.NewTextPart("Go is a programming language.")),
			},
			expected: []interface{}{
				// For simple text responses, we expect a ResponseMessage with Content.
				client.ChatMessage{Role: "assistant", Content: "Go is a programming language."},
			},
			expectErr: false,
		},
		{
			name: "Full tool call and response sequence",
			input: []*ai.Message{
				// 1. User asks to use a tool
				ai.NewUserMessage(ai.NewTextPart("what is a gablorken of 2 over 3.5?")),
				// 2. Model responds with a request to call the tool
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "gablorken",
							Input: map[string]any{"Value": 2, "Over": 3.5},
							Ref:   "tool-call-id-123", // The crucial ID
						}),
					},
				},
				// 3. The tool is executed and the result is sent back
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "gablorken",
							Output: map[string]any{"result": 11.31},
							Ref:    "tool-call-id-123", // The same ID is passed back
						}),
					},
				},
			},
			expected: []interface{}{
				// User message
				client.ChatMessage{Role: "user", Content: "what is a gablorken of 2 over 3.5?"},
				// Assistant message with the tool call request
				client.ResponseMessage{
					Role: "assistant",
					// Add an empty string content field to satisfy strict models like qwen.
					Content: new(string),
					ToolCalls: []client.ToolCall{
						{
							ID:   "tool-call-id-123",
							Type: "function",
							Function: client.FunctionToCall{
								Name:      "gablorken",
								Arguments: `{"Over":3.5,"Value":2}`,
							},
						},
					},
				},
				// Tool result message
				client.ToolMessage{
					Role:       "tool",
					Content:    `{"result":11.31}`,
					ToolCallID: "tool-call-id-123",
				},
			},
			expectErr: false,
		},
		{
			name: "Tool with only string parameters",
			input: []*ai.Message{
				ai.NewUserMessage(ai.NewTextPart("lookup user Jane Doe")),
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "lookupUser",
							Input: map[string]any{"firstName": "Jane", "lastName": "Doe"},
							Ref:   "lookup-id-456",
						}),
					},
				},
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "lookupUser",
							Output: map[string]any{"status": "found"},
							Ref:    "lookup-id-456",
						}),
					},
				},
			},
			expected: []interface{}{
				client.ChatMessage{Role: "user", Content: "lookup user Jane Doe"},
				client.ResponseMessage{
					Role:    "assistant",
					Content: new(string),
					ToolCalls: []client.ToolCall{{
						ID:   "lookup-id-456",
						Type: "function",
						Function: client.FunctionToCall{
							Name:      "lookupUser",
							Arguments: `{"firstName":"Jane","lastName":"Doe"}`,
						},
					}},
				},
				client.ToolMessage{Role: "tool", Content: `{"status":"found"}`, ToolCallID: "lookup-id-456"},
			},
			expectErr: false,
		},
		{
			name: "Tool with mixed primitive types",
			input: []*ai.Message{
				ai.NewUserMessage(ai.NewTextPart("create an alert for 'server-down' with priority 1 and silent false")),
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name: "createAlert",
							Input: map[string]any{
								"name":     "server-down",
								"priority": 1,
								"silent":   false,
							},
							Ref: "alert-id-789",
						}),
					},
				},
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "createAlert",
							Output: map[string]any{"alertId": "xyz-123", "status": "created"},
							Ref:    "alert-id-789",
						}),
					},
				},
			},
			expected: []interface{}{
				client.ChatMessage{Role: "user", Content: "create an alert for 'server-down' with priority 1 and silent false"},
				client.ResponseMessage{
					Role:    "assistant",
					Content: new(string),
					ToolCalls: []client.ToolCall{{
						ID:   "alert-id-789",
						Type: "function",
						Function: client.FunctionToCall{
							Name:      "createAlert",
							Arguments: `{"name":"server-down","priority":1,"silent":false}`,
						},
					}},
				},
				client.ToolMessage{Role: "tool", Content: `{"alertId":"xyz-123","status":"created"}`, ToolCallID: "alert-id-789"},
			},
			expectErr: false,
		},
		{
			name: "Tool with nested object parameter",
			input: []*ai.Message{
				ai.NewUserMessage(ai.NewTextPart("update user config with theme dark and notifications enabled")),
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name: "updateConfig",
							Input: map[string]any{
								"userId": "user-1",
								"config": map[string]any{
									"theme":         "dark",
									"notifications": true,
								},
							},
							Ref: "config-id-abc",
						}),
					},
				},
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "updateConfig",
							Output: map[string]any{"status": "success"},
							Ref:    "config-id-abc",
						}),
					},
				},
			},
			expected: []interface{}{
				client.ChatMessage{Role: "user", Content: "update user config with theme dark and notifications enabled"},
				client.ResponseMessage{
					Role:    "assistant",
					Content: new(string),
					ToolCalls: []client.ToolCall{{
						ID:   "config-id-abc",
						Type: "function",
						Function: client.FunctionToCall{
							Name:      "updateConfig",
							Arguments: `{"config":{"notifications":true,"theme":"dark"},"userId":"user-1"}`,
						},
					}},
				},
				client.ToolMessage{Role: "tool", Content: `{"status":"success"}`, ToolCallID: "config-id-abc"},
			},
			expectErr: false,
		},
		{
			name: "Tool with array of strings",
			input: []*ai.Message{
				ai.NewUserMessage(ai.NewTextPart("add tags 'urgent' and 'review' to ticket 123")),
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "addTags",
							Input: map[string]any{"ticketId": 123, "tags": []string{"urgent", "review"}},
							Ref:   "tags-id-def",
						}),
					},
				},
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "addTags",
							Output: map[string]any{"updated": true},
							Ref:    "tags-id-def",
						}),
					},
				},
			},
			expected: []interface{}{
				client.ChatMessage{Role: "user", Content: "add tags 'urgent' and 'review' to ticket 123"},
				client.ResponseMessage{
					Role:    "assistant",
					Content: new(string),
					ToolCalls: []client.ToolCall{{
						ID:   "tags-id-def",
						Type: "function",
						Function: client.FunctionToCall{
							Name:      "addTags",
							Arguments: `{"tags":["urgent","review"],"ticketId":123}`,
						},
					}},
				},
				client.ToolMessage{Role: "tool", Content: `{"updated":true}`, ToolCallID: "tags-id-def"},
			},
			expectErr: false,
		},
		{
			name: "Multiple tool calls in a single turn",
			input: []*ai.Message{
				ai.NewUserMessage(ai.NewTextPart("Find user 'jdoe' and get their last login.")),
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name: "findUser", Input: map[string]any{"username": "jdoe"}, Ref: "multi-id-1",
						}),
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name: "getLastLogin", Input: map[string]any{"userId": "user-456"}, Ref: "multi-id-2",
						}),
					},
				},
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name: "findUser", Output: map[string]any{"userId": "user-456"}, Ref: "multi-id-1",
						}),
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name: "getLastLogin", Output: map[string]any{"timestamp": "2025-07-28T14:30:00Z"}, Ref: "multi-id-2",
						}),
					},
				},
			},
			expected: []interface{}{
				client.ChatMessage{Role: "user", Content: "Find user 'jdoe' and get their last login."},
				client.ResponseMessage{
					Role:    "assistant",
					Content: new(string),
					ToolCalls: []client.ToolCall{
						{
							ID: "multi-id-1", Type: "function",
							Function: client.FunctionToCall{Name: "findUser", Arguments: `{"username":"jdoe"}`},
						},
						{
							ID: "multi-id-2", Type: "function",
							Function: client.FunctionToCall{Name: "getLastLogin", Arguments: `{"userId":"user-456"}`},
						},
					},
				},
				client.ToolMessage{Role: "tool", Content: `{"userId":"user-456"}`, ToolCallID: "multi-id-1"},
				client.ToolMessage{Role: "tool", Content: `{"timestamp":"2025-07-28T14:30:00Z"}`, ToolCallID: "multi-id-2"},
			},
			expectErr: false,
		},
		{
			name: "Model message with both tool request and text content",
			input: []*ai.Message{
				{
					Role: ai.RoleModel,
					Content: []*ai.Part{
						// This part makes msg.Text() non-empty
						ai.NewTextPart("I will now call the tool."),
						// This part makes the toolCalls slice non-empty
						ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "someTool",
							Input: map[string]any{"arg": "value"},
							Ref:   "combo-id-1",
						}),
					},
				},
			},
			expected: []interface{}{
				// The if/else if logic should prioritize the tool call
				// and ignore the text part, producing only one message.
				client.ResponseMessage{
					Role:    "assistant",
					Content: new(string),
					ToolCalls: []client.ToolCall{
						{
							ID:   "combo-id-1",
							Type: "function",
							Function: client.FunctionToCall{
								Name:      "someTool",
								Arguments: `{"arg":"value"}`,
							},
						},
					},
				},
			},
			expectErr: false,
		},
		{
			name: "Error on unmarshalable tool response output",
			input: []*ai.Message{
				{
					Role: ai.RoleTool,
					Content: []*ai.Part{
						ai.NewToolResponsePart(&ai.ToolResponse{
							Name:   "badTool",
							Ref:    "bad-id-1",
							Output: make(chan int), // This type cannot be marshaled to JSON
						}),
					},
				},
			},
			expected:  nil,
			expectErr: true,
		},
	}

	// Run the test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Use require for assertions, which stops the test on failure.
			r := require.New(t)

			got, err := toClientMessages(tc.input)

			if tc.expectErr {
				r.Error(err)
				return // Test ends here if an error is expected.
			}

			r.NoError(err)
			r.NotNil(got)
			r.Len(got, len(tc.expected))

			// Compare each message, handling the special case for tool calls.
			for i := range tc.expected {
				expectedMsg := tc.expected[i]
				gotMsg := got[i]

				// The ResponseMessage and ChatMessage now need to be handled carefully
				// since they are both used for assistant roles.
				switch expected := expectedMsg.(type) {
				case client.ResponseMessage:
					got, ok := gotMsg.(client.ResponseMessage)
					r.True(ok, "expected message type client.ResponseMessage, but got %T", gotMsg)
					r.Equal(expected.Role, got.Role)
					if expected.Content != nil || got.Content != nil {
						r.NotNil(expected.Content)
						r.NotNil(got.Content)
						r.Equal(*expected.Content, *got.Content)
					}
					r.Len(got.ToolCalls, len(expected.ToolCalls))
					for j, expectedCall := range expected.ToolCalls {
						gotCall := got.ToolCalls[j]
						r.Equal(expectedCall.ID, gotCall.ID)
						r.Equal(expectedCall.Type, gotCall.Type)
						r.Equal(expectedCall.Function.Name, gotCall.Function.Name)
						r.JSONEq(expectedCall.Function.Arguments, gotCall.Function.Arguments)
					}
				case client.ChatMessage:
					got, ok := gotMsg.(client.ChatMessage)
					r.True(ok, "expected message type client.ChatMessage, but got %T", gotMsg)
					r.Equal(expected, got)
				case client.ToolMessage:
					got, ok := gotMsg.(client.ToolMessage)
					r.True(ok, "expected message type client.ToolMessage, but got %T", gotMsg)
					r.Equal(expected, got)
				default:
					r.Failf("unhandled message type", "type: %T", expected)
				}
			}
		})
	}
}
