// ai.go
package main

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

// CompletionUsageLike — компактный аналог usage из OpenAI для логирования в БД.
type CompletionUsageLike struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// AIRequestOptions описывает параметры вызова Chat Completion.
type AIRequestOptions struct {
	Model       string
	MaxTokens   param.Opt[int64]
	Temperature param.Opt[float64]
	Messages    []openai.ChatCompletionMessageParamUnion
	APILogID    *APILogID
}

// callChatCompletion вызывает OpenAI Chat Completion с единым логированием.
func (app *App) callChatCompletion(ctx context.Context, prompt string, opts AIRequestOptions) (string, *CompletionUsageLike, error) {
	trimmedPrompt := strings.TrimSpace(prompt)
	if trimmedPrompt == "" && len(opts.Messages) == 0 {
		return "", nil, errors.New("empty prompt")
	}

	model := strings.TrimSpace(opts.Model)
	if model == "" {
		model = app.Config.ModelName
	}

	messages := opts.Messages
	if len(messages) == 0 {
		messages = []openai.ChatCompletionMessageParamUnion{openai.UserMessage(trimmedPrompt)}
	}

	var aiID AILogID
	if trimmedPrompt != "" || len(messages) > 0 {
		if app.Store != nil {
			var startErr error
			aiID, startErr = app.Store.CreateAILog(ctx, opts.APILogID, model, preview512(trimmedPrompt))
			if startErr != nil {
				logWarn("aiLogStart failed", map[string]interface{}{"error": startErr.Error(), "model": model})
				aiID = 0
			}
		} else {
			logWarn("store not initialized — AI log disabled", map[string]interface{}{"model": model})
		}
	}
	started := time.Now()

	params := openai.ChatCompletionNewParams{
		Model:       openai.ChatModel(model),
		Messages:    messages,
		MaxTokens:   opts.MaxTokens,
		Temperature: opts.Temperature,
	}

	resp, err := app.Config.AIClient.Chat.Completions.New(ctx, params)
	if err != nil {
		if app.Store != nil && aiID != 0 {
			if err := app.Store.UpdateAILog(ctx, aiID, "", err.Error(), nil, time.Since(started)); err != nil {
				logWarn("aiLogFinish failed", map[string]interface{}{"error": err.Error(), "model": model})
			}
		}
		return "", nil, err
	}
	if len(resp.Choices) == 0 {
		if app.Store != nil && aiID != 0 {
			if err := app.Store.UpdateAILog(ctx, aiID, "", "no choices from AI", nil, time.Since(started)); err != nil {
				logWarn("aiLogFinish failed", map[string]interface{}{"error": err.Error(), "model": model})
			}
		}
		return "", nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)

	usageStruct := CompletionUsageLike{
		PromptTokens:     int(resp.Usage.PromptTokens),
		CompletionTokens: int(resp.Usage.CompletionTokens),
		TotalTokens:      int(resp.Usage.TotalTokens),
	}
	var usage *CompletionUsageLike
	if usageStruct.PromptTokens != 0 || usageStruct.CompletionTokens != 0 || usageStruct.TotalTokens != 0 {
		usage = &usageStruct
	}

	if app.Store != nil && aiID != 0 {
		if err := app.Store.UpdateAILog(ctx, aiID, raw, "", usage, time.Since(started)); err != nil {
			logWarn("aiLogFinish failed", map[string]interface{}{"error": err.Error(), "model": model})
		}
	}
	return raw, usage, nil
}
