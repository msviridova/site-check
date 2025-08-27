package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
)

// Глобальный клиент: инициализируется в main.go
var aiClient openai.Client

// CompletionUsageLike — упрощённая структура для логирования usage.
type CompletionUsageLike struct {
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
}

// toUsageLike приводит openai.CompletionUsage к нашей структуре.
func toUsageLike(u *openai.CompletionUsage) *CompletionUsageLike {
	if u == nil {
		return nil
	}
	return &CompletionUsageLike{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
}

// summarizeWithAI вызывает OpenAI и возвращает summary, keywords, negative_keywords.
func summarizeWithAI(ctx context.Context, text string) (string, []string, []string, error) {
	preview := strings.TrimSpace(text)
	if len(preview) > 512 {
		preview = preview[:512]
	}

	var apiRef *APILogID = nil
	start := time.Now()
	aiID, _ := aiLogStart(ctx, apiRef, modelName, preview)

	if len(text) > 4000 {
		text = text[:4000]
	}

	cctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	prompt := `Ты — сервис классификации сайтов.

1) Кратко, одной деловой фразой по-русски опиши тематику сайта (сфера/услуга/товар и, если явно есть, город/бренд).
   Не добавляй лишних слов, без пояснений, без ссылок.

2) Сгенерируй список ключевых слов и фраз для запуска рекламы в Яндекс.Директ (30–40 штук, только по этому контенту).

3) Сформируй список минус-слов (30–50), чтобы отсеять нерелевантные запросы.

Верни СТРОГО валидный JSON ровно такой структуры (без пояснений снаружи):
{
  "summary": "краткое описание одной фразой",
  "keywords": ["...", "..."],
  "negative_keywords": ["...", "..."]
}

Контент сайта:
` + text

	// Вызов chat-completion
	resp, err := aiClient.Chat.Completions.New(cctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		MaxTokens:   openai.Int(800),
		Temperature: openai.Float(0.2),
		Seed:        openai.Int(42),
	})
	if err != nil {
		aiLogFinish(ctx, aiID, "", err.Error(), nil, time.Since(start))
		return "", nil, nil, err
	}
	if len(resp.Choices) == 0 {
		aiLogFinish(ctx, aiID, "", "no choices from AI", nil, time.Since(start))
		return "", nil, nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	if raw == "" {
		aiLogFinish(ctx, aiID, "", "empty AI response", nil, time.Since(start))
		return "", nil, nil, errors.New("empty AI response")
	}

	// usage
	usageLike := toUsageLike(&resp.Usage)

	// Парсим JSON
	var tmp struct {
		Summary          string   `json:"summary"`
		Keywords         []string `json:"keywords"`
		NegativeKeywords []string `json:"negative_keywords"`
	}
	if jerr := json.Unmarshal([]byte(raw), &tmp); jerr != nil {
		aiLogFinish(ctx, aiID, raw, "AI returned non-JSON: "+jerr.Error(), usageLike, time.Since(start))
		return raw, nil, nil, nil
	}

	aiLogFinish(ctx, aiID, raw, "", usageLike, time.Since(start))
	return strings.TrimSpace(tmp.Summary), tmp.Keywords, tmp.NegativeKeywords, nil
}
