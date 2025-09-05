// ai_creatives.go
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

// generateAllTextCreatives — генерирует keywords, negatives и ads одним промптом из БД
// Ожидается, что в БД лежит ключ 'creative_text_all' с плейсхолдером {website_text}
func generateAllTextCreatives(ctx context.Context, siteText string) (*TextCreatives, error) {
	const promptKey = "creative_text_all"

	p, err := getPrompt(db, promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := strings.ReplaceAll(p.Text, "{website_text}", siteText)

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

// generateKeywords — берёт промпт из БД по ключу 'creative_text_keywords'
// Ожидаемые варианты ответа:
// 1) {"keywords": { "категория1":[], "категория2": [] }}
// 2) ["фраза 1","фраза 2",...]
// 3) текст построчно
func generateKeywords(ctx context.Context, siteText string) ([]string, error) {
	const promptKey = "creative_text_keywords"

	p, err := getPrompt(db, promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := strings.ReplaceAll(p.Text, "{website_text}", siteText)

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
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

	// вариант {"keywords": {...}}
	var obj struct {
		Keywords map[string][]string `json:"keywords"`
	}
	if jerr := json.Unmarshal([]byte(raw), &obj); jerr == nil && obj.Keywords != nil {
		var out []string
		for _, arr := range obj.Keywords {
			out = append(out, arr...)
		}
		return dedup(out), nil
	}

	// вариант ["...","..."]
	var flat []string
	if jerr2 := json.Unmarshal([]byte(raw), &flat); jerr2 == nil {
		return dedup(flat), nil
	}

	// fallback — построчно
	lines := strings.Split(raw, "\n")
	var out []string
	for _, l := range lines {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return dedup(out), nil
}

// generateNegatives — берёт промпт из БД по ключу 'creative_text_negatives'
// Ожидаемые варианты аналогичны generateKeywords
func generateNegatives(ctx context.Context, siteText string) ([]string, error) {
	const promptKey = "creative_text_negatives"

	p, err := getPrompt(db, promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := strings.ReplaceAll(p.Text, "{website_text}", siteText)

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
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

	// вариант {"negatives": {...}}
	var obj struct {
		Negatives map[string][]string `json:"negatives"`
	}
	if jerr := json.Unmarshal([]byte(raw), &obj); jerr == nil && obj.Negatives != nil {
		var out []string
		for _, arr := range obj.Negatives {
			out = append(out, arr...)
		}
		return dedup(out), nil
	}

	// вариант ["...","..."]
	var flat []string
	if jerr2 := json.Unmarshal([]byte(raw), &flat); jerr2 == nil {
		return dedup(flat), nil
	}

	// fallback — построчно
	lines := strings.Split(raw, "\n")
	var out []string
	for _, l := range lines {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return dedup(out), nil
}

// generateAds — берёт промпт из БД по ключу 'creative_text_ads'
// Ожидается JSON-массив []AdBlock
func generateAds(ctx context.Context, siteText string) ([]AdBlock, error) {
	const promptKey = "creative_text_ads"

	p, err := getPrompt(db, promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := strings.ReplaceAll(p.Text, "{website_text}", siteText)

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
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

// generateGraphic — берёт промпт из БД по ключу 'creative_graphic'
// Поддержанные плейсхолдеры в тексте промпта:
// {website_text}, {site_url}, {goal}, {audience}, {geo}, {offer_constraints}, {brand_overrides}
func generateGraphic(ctx context.Context, siteURL, siteText string, opts GraphicInputOpts) (*GraphicPlan, error) {
	const promptKey = "creative_graphic"

	p, err := getPrompt(db, promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}

	pp := p.Text
	pp = strings.ReplaceAll(pp, "{website_text}", siteText)
	pp = strings.ReplaceAll(pp, "{site_url}", siteURL)
	pp = strings.ReplaceAll(pp, "{goal}", nz(opts.Goal))
	pp = strings.ReplaceAll(pp, "{audience}", nz(opts.Audience))
	pp = strings.ReplaceAll(pp, "{geo}", nz(opts.Geo))
	pp = strings.ReplaceAll(pp, "{offer_constraints}", nz(opts.OfferConstraints))
	pp = strings.ReplaceAll(pp, "{brand_overrides}", nz(opts.BrandOverrides))

	resp, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(pp),
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
