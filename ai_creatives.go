package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/openai/openai-go/v2"
)

// ====== ТЕКСТОВЫЕ КРЕАТИВЫ ======

// TextCreatives содержит все типы текстовых креативов
type TextCreatives struct {
	Keywords  []string  `json:"keywords"`
	Negatives []string  `json:"negatives"`
	Ads       []AdBlock `json:"ads"`
}

// generateAllTextCreatives генерирует все типы текстовых креативов за один запрос к OpenAI
func generateAllTextCreatives(ctx context.Context, siteText string) (*TextCreatives, error) {
	prompt := `Ты — ИИ-агент для генерации рекламных материалов для Яндекс.Директ.
Анализируй только предоставленный текст сайта и сгенерируй все необходимые материалы.

**Контент сайта:**
` + siteText + `

Сформируй СТРОГО валидный JSON со всеми типами креативов:

{
  "keywords": [
    "ключевое слово 1",
    "ключевое слово 2"
  ],
  "negatives": [
    "минус-слово 1",
    "минус-слово 2"
  ],
  "ads": [
    {
      "id": "AD1",
      "header": "Заголовок ≤56 символов",
      "text": "Текст ≤81 символ, с CTA",
      "links": [
        {"url":"https://...", "title":"≤30", "desc":"≤60"},
        {"url":"https://...", "title":"≤30", "desc":"≤60"}
      ],
      "details": ["Уточнение1", "Уточнение2"]
    }
  ]
}

Требования:
1. **Keywords**: 30-40 релевантных ключевых слов и фраз для контекстной рекламы. Без "дешево/лучший/топ", без орфографических ошибок.
2. **Negatives**: 30-50 минус-слов для отсечения нерелевантного трафика (бесплатные, вакансии, конкуренты, нерелевантные запросы).
3. **Ads**: 5 рекламных объявлений с соблюдением лимитов символов Яндекс.Директ.

Если данных недостаточно для какого-то поля — оставь пустой массив или пропусти необязательные поля.`

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		MaxTokens:   openai.Int(2500),
		Temperature: openai.Float(0.3),
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)

	var result TextCreatives
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return nil, err
	}

	// Дедупликация ключевых слов и минус-слов
	result.Keywords = dedup(result.Keywords)
	result.Negatives = dedup(result.Negatives)

	return &result, nil
}

func generateKeywords(ctx context.Context, siteText string) ([]string, error) {
	p := strings.ReplaceAll(promptKeywords, "{website_text}", siteText)

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(p),
		},
		MaxTokens:   openai.Int(1000),
		Temperature: openai.Float(0.2),
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)

	// ожидаем JSON вида {"keywords": { "общие":[], "продукты_услуги":[] ... }}
	var tmp struct {
		Keywords map[string][]string `json:"keywords"`
	}
	if jerr := json.Unmarshal([]byte(raw), &tmp); jerr == nil {
		var out []string
		for _, arr := range tmp.Keywords {
			out = append(out, arr...)
		}
		return dedup(out), nil
	}

	// fallback: массив строк
	var flat []string
	if jerr2 := json.Unmarshal([]byte(raw), &flat); jerr2 == nil {
		return dedup(flat), nil
	}

	// fallback: разделение по строкам
	lines := strings.Split(raw, "\n")
	var out []string
	for _, l := range lines {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return dedup(out), nil
}

func generateNegatives(ctx context.Context, siteText string) ([]string, error) {
	p := strings.ReplaceAll(promptNegatives, "{website_text}", siteText)

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(p),
		},
		MaxTokens:   openai.Int(1000),
		Temperature: openai.Float(0.2),
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)

	var tmp struct {
		Negatives map[string][]string `json:"negatives"`
	}
	if jerr := json.Unmarshal([]byte(raw), &tmp); jerr == nil {
		var out []string
		for _, arr := range tmp.Negatives {
			out = append(out, arr...)
		}
		return dedup(out), nil
	}

	// fallback: массив строк
	var flat []string
	if jerr2 := json.Unmarshal([]byte(raw), &flat); jerr2 == nil {
		return dedup(flat), nil
	}

	// fallback: разделение по строкам
	lines := strings.Split(raw, "\n")
	var out []string
	for _, l := range lines {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return dedup(out), nil
}

func generateAds(ctx context.Context, siteText string) ([]AdBlock, error) {
	p := strings.ReplaceAll(promptAds, "{website_text}", siteText)

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(p),
		},
		MaxTokens:   openai.Int(1200),
		Temperature: openai.Float(0.3),
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)

	var ads []AdBlock
	if err := json.Unmarshal([]byte(raw), &ads); err != nil {
		return nil, err
	}
	return ads, nil
}

// ====== ГРАФИКА ======

func generateGraphic(ctx context.Context, siteURL, siteText string, opts GraphicInputOpts) (*GraphicPlan, error) {
	p := strings.ReplaceAll(promptGraphic, "{website_text}", siteText)
	p = strings.ReplaceAll(p, "{site_url}", siteURL)
	p = strings.ReplaceAll(p, "{goal}", nz(opts.Goal))
	p = strings.ReplaceAll(p, "{audience}", nz(opts.Audience))
	p = strings.ReplaceAll(p, "{geo}", nz(opts.Geo))
	p = strings.ReplaceAll(p, "{offer_constraints}", nz(opts.OfferConstraints))
	p = strings.ReplaceAll(p, "{brand_overrides}", nz(opts.BrandOverrides))

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(p),
		},
		MaxTokens:   openai.Int(2000),
		Temperature: openai.Float(0.5),
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)

	var gp GraphicPlan
	if err := json.Unmarshal([]byte(raw), &gp); err != nil {
		return nil, err
	}
	return &gp, nil
}

// ====== Утилиты ======

func dedup(xs []string) []string {
	seen := make(map[string]struct{}, len(xs))
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		x = strings.TrimSpace(x)
		if x == "" {
			continue
		}
		if _, ok := seen[x]; ok {
			continue
		}
		seen[x] = struct{}{}
		out = append(out, x)
	}
	return out
}

func nz(s string) string {
	if strings.TrimSpace(s) == "" {
		return ""
	}
	return s
}
