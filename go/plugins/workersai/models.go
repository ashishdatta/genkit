package workersai

import "github.com/firebase/genkit/go/ai"

const (
	mistralSmall3124BInstruct     = "@cf/mistralai/mistral-small-3.1-24b-instruct"
	metaLlama3370bInstructFp8Fast = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
)

var (
	workersAIModels = []string{
		mistralSmall3124BInstruct,
		metaLlama3370bInstructFp8Fast,
	}

	supportedWorkersAIModels = map[string]ai.ModelInfo{
		mistralSmall3124BInstruct: {
			Label: mistralSmall3124BInstruct,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      true,
				ToolChoice: false,
				SystemRole: true,
				Media:      true,
			},
		},

		metaLlama3370bInstructFp8Fast: {
			Label: metaLlama3370bInstructFp8Fast,
			Supports: &ai.ModelSupports{
				Multiturn:  true,
				Tools:      true,
				ToolChoice: false,
				SystemRole: true,
				Media:      true,
			},
		},
	}
)
