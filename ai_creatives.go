// ai_creatives.go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
)

// ---- лёгкий кэш для текстов промптов (по ключу/локали/версии) ----

func promptCacheKey(key, locale string, version int) string {
	return key + "||" + locale + "||" + strconv.Itoa(version)
}

func (app *App) getPromptCached(key, locale string, version int) (*Prompt, error) {
	if app == nil {
		return nil, errors.New("app is nil")
	}
	ttl := app.promptCache.ttl
	k := promptCacheKey(key, locale, version)

	app.promptCache.mu.RLock()
	if entry, ok := app.promptCache.m[k]; ok && entry.prompt != nil {
		if ttl <= 0 || time.Now().Before(entry.expires) {
			app.promptCache.mu.RUnlock()
			return entry.prompt, nil
		}
	}
	app.promptCache.mu.RUnlock()

	if app.Store == nil {
		return nil, errors.New("store is not initialized")
	}

	p, err := app.Store.GetPrompt(key, locale, version)
	if err != nil {
		return nil, err
	}

	app.promptCache.mu.Lock()
	expires := time.Time{}
	if ttl > 0 {
		expires = time.Now().Add(ttl)
	}
	app.promptCache.m[k] = cachedPrompt{prompt: p, expires: expires}
	app.promptCache.mu.Unlock()

	return p, nil
}

// ====== ТЕКСТОВЫЕ КРЕАТИВЫ ======

// TextCreatives содержит все типы текстовых креативов
type TextCreatives struct {
	Keywords  []string  `json:"keywords"`
	Negatives []string  `json:"negatives"`
	Ads       []AdBlock `json:"ads"`
}

// generateAllTextCreatives — генерирует keywords, negatives и ads одним промптом из БД
// Ожидается, что в БД лежит ключ 'creative_text_all' с плейсхолдером {website_text}
func (app *App) generateAllTextCreatives(ctx context.Context, siteText string) (*TextCreatives, error) {
	const promptKey = "creative_text_all"

	p, err := app.getPromptCached(promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := applyPlaceholders(p.Text, map[string]string{
		"{website_text}": siteText,
	})

	raw, _, err := app.callChatCompletion(ctx, prompt, AIRequestOptions{
		MaxTokens:   openai.Int(2500),
		Temperature: openai.Float(0.3),
	})
	if err != nil {
		return nil, err
	}

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
func (app *App) generateKeywords(ctx context.Context, siteText string) ([]string, error) {
	const promptKey = "creative_text_keywords"

	p, err := app.getPromptCached(promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := applyPlaceholders(p.Text, map[string]string{
		"{website_text}": siteText,
	})

	raw, _, err := app.callChatCompletion(ctx, prompt, AIRequestOptions{
		MaxTokens:   openai.Int(1000),
		Temperature: openai.Float(0.2),
	})
	if err != nil {
		return nil, err
	}

	// вариант {"keywords": {...}}
	if nested, ok := parseNestedStringList(raw, "keywords"); ok {
		return dedup(nested), nil
	}

	// вариант ["...","..."]
	var flat []string
	if jerr := json.Unmarshal([]byte(raw), &flat); jerr == nil {
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
func (app *App) generateNegatives(ctx context.Context, siteText string) ([]string, error) {
	const promptKey = "creative_text_negatives"

	p, err := app.getPromptCached(promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := applyPlaceholders(p.Text, map[string]string{
		"{website_text}": siteText,
	})

	raw, _, err := app.callChatCompletion(ctx, prompt, AIRequestOptions{
		MaxTokens:   openai.Int(1000),
		Temperature: openai.Float(0.2),
	})
	if err != nil {
		return nil, err
	}

	// вариант {"negatives": {...}}
	if nested, ok := parseNestedStringList(raw, "negatives"); ok {
		return dedup(nested), nil
	}

	// вариант ["...","..."]
	var flat []string
	if jerr := json.Unmarshal([]byte(raw), &flat); jerr == nil {
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
func (app *App) generateAds(ctx context.Context, siteText string) ([]AdBlock, error) {
	const promptKey = "creative_text_ads"

	p, err := app.getPromptCached(promptKey, "ru", 0)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}
	prompt := applyPlaceholders(p.Text, map[string]string{
		"{website_text}": siteText,
	})

	raw, _, err := app.callChatCompletion(ctx, prompt, AIRequestOptions{
		MaxTokens:   openai.Int(1200),
		Temperature: openai.Float(0.3),
	})
	if err != nil {
		return nil, err
	}

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
func (app *App) generateGraphic(ctx context.Context, siteURL, siteText string, opts GraphicInputOpts) (*GraphicPlan, error) {
	const (
		promptKey   = "creative_graphic"
		locale      = "ru"
		version     = 0    // 0 = взять актуальную активную версию
		maxSiteText = 8000 // жёсткая подрезка, чтобы не раздувать запрос
	)

	p, err := app.getPromptCached(promptKey, locale, version)
	if err != nil {
		return nil, errors.New("prompt not found: " + promptKey)
	}

	siteText = strings.TrimSpace(siteText)
	if siteText == "" {
		return nil, errors.New("empty website_text")
	}
	if max := app.Config.GraphicSiteTextMax; max > 0 && len(siteText) > max {
		siteText = siteText[:max]
	}
	siteURL = strings.TrimSpace(siteURL)

	pp := applyPlaceholders(p.Text, map[string]string{
		"{website_text}":      strings.TrimSpace(siteText),
		"{site_url}":          strings.TrimSpace(siteURL),
		"{goal}":              strings.TrimSpace(opts.Goal),
		"{audience}":          strings.TrimSpace(opts.Audience),
		"{geo}":               strings.TrimSpace(opts.Geo),
		"{offer_constraints}": strings.TrimSpace(opts.OfferConstraints),
		"{brand_overrides}":   strings.TrimSpace(opts.BrandOverrides),
	})

	cctx, cancel := context.WithTimeout(ctx, app.Config.AITimeout)
	defer cancel()

	raw, _, err := app.callChatCompletion(cctx, pp, AIRequestOptions{
		MaxTokens:   openai.Int(1300),
		Temperature: openai.Float(0.45),
	})
	if err != nil {
		return nil, err
	}

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

func parseNestedStringList(raw, key string) ([]string, bool) {
	var obj map[string]map[string][]string
	if err := json.Unmarshal([]byte(raw), &obj); err != nil {
		return nil, false
	}
	nested, ok := obj[key]
	if !ok || nested == nil {
		return nil, false
	}
	var out []string
	for _, values := range nested {
		out = append(out, values...)
	}
	return out, true
}
