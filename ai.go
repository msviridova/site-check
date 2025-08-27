// ai.go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
)

var aiClient openai.Client

// Цвета от ИИ (что парсим из JSON)
type AIColors struct {
	Main            []string `json:"main"`
	Additional      []string `json:"additional"`
	Background      string   `json:"background"`
	AccentPrimary   string   `json:"accent_primary"`
	AccentSecondary string   `json:"accent_secondary"`
}

// «прокладка» под usage OpenAI для логов БД
type CompletionUsageLike struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

func toUsageLike(u *openai.CompletionUsage) *CompletionUsageLike {
	if u == nil {
		return nil
	}
	// у v2 это int64 — приводим к int
	return &CompletionUsageLike{
		PromptTokens:     int(u.PromptTokens),
		CompletionTokens: int(u.CompletionTokens),
		TotalTokens:      int(u.TotalTokens),
	}
}

// normalizeHex: приводит "#aabbcc" / "aabbcc" / "#ABC" -> "#AABBCC"/"#ABC"
func normalizeHex(s string) string {
	x := strings.ToUpper(strings.TrimSpace(s))
	if x == "" {
		return ""
	}
	if !strings.HasPrefix(x, "#") {
		x = "#" + x
	}
	// допускаем #RGB или #RRGGBB
	if len(x) == 4 || len(x) == 7 {
		return x
	}
	// всё остальное — выбрасываем
	return ""
}

// summarizeWithAI вызывает OpenAI и возвращает summary, keywords, negative_keywords, colors.
func summarizeWithAI(ctx context.Context, text string) (string, []string, []string, *AIColors, error) {
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

	// ⇩⇩⇩ новый промт с детальным анализом цветов
	prompt := `Ты — сервис классификации сайтов.

1) Кратко, одной деловой фразой по-русски опиши тематику сайта (сфера/услуга/товар и, если явно есть, город/бренд).
   Не добавляй лишних слов, без пояснений, без ссылок.

2) Сгенерируй список ключевых слов и фраз для запуска рекламы в Яндекс.Директ (30–40 штук, только по этому контенту).

3) Сформируй список минус-слов (30–50), чтобы отсеять нерелевантные запросы.

4) Проанализируй фирменные цвета сайта и верни палитру:
   - "main": 2–4 основных цвета (HEX)
   - "additional": 2–6 дополнительных цветов (HEX), если есть
   - "background": основной цвет фона (HEX), если можно определить
   - "accent_primary": цвет основных акцентов/кнопок/хедера (HEX), если видно
   - "accent_secondary": дополнительный акцент (HEX), если есть

Правила по цветам:
- Всегда возвращай HEX (#RRGGBB или #RGB), без именованных цветов.
- Если не уверен — оставь пустую строку или пустой массив.
- Если явного цвета нет в тексте/разметке — не выдумывай.

Верни СТРОГО валидный JSON ровно такой структуры (без пояснений снаружи):
{
  "summary": "краткое описание одной фразой",
  "keywords": ["...", "..."],
  "negative_keywords": ["...", "..."],
  "colors": {
    "main": ["#...", "#..."],
    "additional": ["#..."],
    "background": "#...",
    "accent_primary": "#...",
    "accent_secondary": "#..."
  }
}

Контент сайта:
` + text

	resp, err := aiClient.Chat.Completions.New(cctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		MaxTokens:   openai.Int(900),
		Temperature: openai.Float(0.2),
		Seed:        openai.Int(42),
	})
	if err != nil {
		aiLogFinish(ctx, aiID, "", err.Error(), nil, time.Since(start))
		return "", nil, nil, nil, err
	}
	if len(resp.Choices) == 0 {
		aiLogFinish(ctx, aiID, "", "no choices from AI", nil, time.Since(start))
		return "", nil, nil, nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	if raw == "" {
		aiLogFinish(ctx, aiID, "", "empty AI response", nil, time.Since(start))
		return "", nil, nil, nil, errors.New("empty AI response")
	}

	usageLike := toUsageLike(&resp.Usage)

	// ожидаем ровно ту структуру
	var tmp struct {
		Summary          string   `json:"summary"`
		Keywords         []string `json:"keywords"`
		NegativeKeywords []string `json:"negative_keywords"`
		Colors           struct {
			Main            []string `json:"main"`
			Additional      []string `json:"additional"`
			Background      string   `json:"background"`
			AccentPrimary   string   `json:"accent_primary"`
			AccentSecondary string   `json:"accent_secondary"`
		} `json:"colors"`
	}

	if jerr := json.Unmarshal([]byte(raw), &tmp); jerr != nil {
		aiLogFinish(ctx, aiID, raw, "AI returned non-JSON: "+jerr.Error(), usageLike, time.Since(start))
		// вернём хотя бы «raw» как summary, остальное пусто
		return raw, nil, nil, nil, nil
	}

	// нормализуем HEX
	norm := func(list []string) []string {
		out := make([]string, 0, len(list))
		for _, c := range list {
			if h := normalizeHex(c); h != "" {
				out = append(out, h)
			}
		}
		return out
	}

	colors := &AIColors{
		Main:            norm(tmp.Colors.Main),
		Additional:      norm(tmp.Colors.Additional),
		Background:      normalizeHex(tmp.Colors.Background),
		AccentPrimary:   normalizeHex(tmp.Colors.AccentPrimary),
		AccentSecondary: normalizeHex(tmp.Colors.AccentSecondary),
	}

	aiLogFinish(ctx, aiID, raw, "", usageLike, time.Since(start))
	return strings.TrimSpace(tmp.Summary), tmp.Keywords, tmp.NegativeKeywords, colors, nil
}
