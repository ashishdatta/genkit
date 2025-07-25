package workersai_test

import (
	"context"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/workersai"
)

func requireEnv(key string) (string, bool) {
	value, ok := os.LookupEnv(key)
	if !ok || value == "" {
		return "", false
	}

	return value, true
}

func TestWorkersAILive(t *testing.T) {
	apiToken, ok := requireEnv("CLOUDFLARE_API_TOKEN")
	if !ok {
		t.Skip("no CLOUDFLARE_API_TOKEN key provided, set CLOUDFLARE_API_TOKEN as an environment variable")
	}

	accountID, ok := requireEnv("CLOUDFLARE_ACCOUNT_ID")
	if !ok {
		t.Skip("no CLOUDFLARE_ACCOUNT_ID provided, set CLOUDFLARE_ACCOUNT_ID as an environment variable")
	}

	ctx := context.Background()

	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&workersai.WorkersAI{APIToken: apiToken, AccountID: accountID}),
		// genkit prefixes the provider to the model name and so the model here is specified with the workersai prefix
		genkit.WithDefaultModel("workersai/mistralai/mistral-small-3.1-24b-instruct"),
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

	// gablorkenTool := genkit.DefineTool(g, "gablorken", "use this tool when the user asks to calculate a gablorken, carefuly inspect the user input to determine which value from the prompt corresponds to the input structure",
	// 	func(ctx *ai.ToolContext, input struct {
	// 		Value int
	// 		Over  float64
	// 	},
	// 	) (float64, error) {
	// 		return math.Pow(float64(input.Value), input.Over), nil
	// 	},
	// )

	// t.Run("tool", func(t *testing.T) {
	// 	resp, err := genkit.Generate(ctx, g,
	// 		ai.WithPrompt("what is a gablorken of 2 over 3.5? use the gablorken tool"),
	// 		ai.WithTools(gablorkenTool))
	// 	if err != nil {
	// 		t.Fatal(err)
	// 	}
	//
	// 	out := resp.Message.Content[0].Text
	// 	const want = "11.31"
	// 	if !strings.Contains(out, want) {
	// 		t.Errorf("got %q, expecting it to contain %q", out, want)
	// 	}
	// })

}
