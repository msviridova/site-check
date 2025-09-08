// handler.go
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
)

func classifyHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("[DEBUG] classifyHandler: Request started, method=%s, url=%s", r.Method, r.URL.Path)

	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	var req classifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("[DEBUG] classifyHandler: JSON decode error: %v", err)
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}
	log.Printf("[DEBUG] classifyHandler: Request decoded, URL=%s", req.URL)

	raw := strings.TrimSpace(req.URL)
	if raw == "" {
		http.Error(w, "url is required", http.StatusBadRequest)
		return
	}
	u, err := url.ParseRequestURI(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		http.Error(w, "invalid url", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 45*time.Second)
	defer cancel()

	// API log
	rawReq, _ := json.Marshal(req)
	apiStart := time.Now()
	apiID, _ := apiLogStart(ctx, "/classify", u.String(), string(rawReq))

	// ── 1) HTML сайта
	html, err := fetchHTML(ctx, u.String())
	if err != nil {
		http.Error(w, "fetch failed: "+err.Error(), http.StatusBadGateway)
		apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "fetch failed: "+err.Error(), time.Since(apiStart))
		return
	}

	// ── 2) Эвристики
	brandHeur := extractBrand(u, html)
	extractedColors := extractColorsHex(html)
	styleHeur := deriveStyleNotes(extractedColors, html)

	// ── 3) Подготовка текста для промпта
	siteText := extractVisibleText(html)
	if len(siteText) > 12000 {
		siteText = siteText[:12000]
	}

	// ── 4) Промпт из БД (без фолбэка)
	const promptKey = "classify"
	log.Printf("[DEBUG] classifyHandler: About to call getPrompt with key=%s, locale=ru, version=0", promptKey)
	p, err := getPrompt(db, promptKey, "ru", 0)
	if err != nil {
		log.Printf("[DEBUG] classifyHandler: getPrompt error: %v", err)
		http.Error(w, "prompt not found: "+promptKey, http.StatusInternalServerError)
		apiLogFinish(ctx, apiID, http.StatusInternalServerError, "", "no prompt in DB", time.Since(apiStart))
		return
	}
	log.Printf("[DEBUG] classifyHandler: getPrompt success, prompt length=%d", len(p.Text))
	prompt := strings.TrimSpace(p.Text)
	if prompt == "" {
		http.Error(w, "prompt is empty: "+promptKey, http.StatusInternalServerError)
		apiLogFinish(ctx, apiID, http.StatusInternalServerError, "", "empty prompt text", time.Since(apiStart))
		return
	}

	// ── 5) Подстановка плейсхолдеров
	if styleHeur == "" {
		styleHeur = "неопределено"
	}
	prompt = strings.ReplaceAll(prompt, "{HEUR_BRAND}", brandHeur)
	prompt = strings.ReplaceAll(prompt, "{HEUR_STYLE}", styleHeur)
	if len(extractedColors) == 0 {
		prompt = strings.ReplaceAll(prompt, "{HEUR_COLORS}", "[]")
	} else {
		prompt = strings.ReplaceAll(prompt, "{HEUR_COLORS}", strings.Join(extractedColors, ", "))
	}
	prompt = strings.ReplaceAll(prompt, "{SITE_TEXT}", siteText)

	// ── 6) Вызов AI
	startAI := time.Now()
	aiID, _ := aiLogStart(ctx, nil, modelName, preview512(prompt))

	respAI, err := aiClient.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: modelName,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		MaxTokens:   openai.Int(900),
		Temperature: openai.Float(0.3),
	})
	if err != nil {
		aiLogFinish(ctx, aiID, "", err.Error(), nil, time.Since(startAI))
		http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
		apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "ai error", time.Since(apiStart))
		return
	}
	if len(respAI.Choices) == 0 {
		aiLogFinish(ctx, aiID, "", "no choices from AI", nil, time.Since(startAI))
		http.Error(w, "AI: no choices", http.StatusBadGateway)
		apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "AI: no choices", time.Since(apiStart))
		return
	}

	rawJSON := strings.TrimSpace(respAI.Choices[0].Message.Content)
	aiLogFinish(ctx, aiID, rawJSON, "", nil, time.Since(startAI))

	// ── 7) Разбор ответа AI → твоя целевая структура
	type aiOut struct {
		Summary             string   `json:"summary"`
		Brand               string   `json:"brand"`
		StyleNotes          string   `json:"style_notes"`
		MainColorsHex       []string `json:"main_colors_hex"`
		AdditionalColorsHex []string `json:"additional_colors_hex"`
		BackgroundColorHex  string   `json:"background_color_hex"`
		AccentPrimaryHex    string   `json:"accent_primary_hex"`
		AccentSecondaryHex  string   `json:"accent_secondary_hex"`
	}
	var out aiOut
	if err := json.Unmarshal([]byte(rawJSON), &out); err != nil {
		http.Error(w, "AI JSON parse error: "+err.Error(), http.StatusBadGateway)
		apiLogFinish(ctx, apiID, http.StatusBadGateway, rawJSON, "ai json parse error", time.Since(apiStart))
		return
	}

	// ── 8) Чистка/дедуп
	uniq := func(xs []string) []string {
		m := make(map[string]struct{}, len(xs))
		out := make([]string, 0, len(xs))
		for _, v := range xs {
			v = strings.TrimSpace(v)
			if v == "" {
				continue
			}
			if _, ok := m[v]; ok {
				continue
			}
			m[v] = struct{}{}
			out = append(out, v)
		}
		return out
	}

	finalBrand := strings.TrimSpace(out.Brand)
	if finalBrand == "" {
		finalBrand = brandHeur
	}
	finalStyle := strings.TrimSpace(out.StyleNotes)
	if finalStyle == "" {
		finalStyle = styleHeur
	}

	resp := classifyResponse{
		Summary:             strings.TrimSpace(out.Summary),
		Lang:                "ru",
		Source:              "ai",
		Brand:               finalBrand,
		StyleNotes:          finalStyle,
		MainColorsHex:       uniq(out.MainColorsHex),
		AdditionalColorsHex: uniq(out.AdditionalColorsHex),
		BackgroundColorHex:  strings.TrimSpace(out.BackgroundColorHex),
		AccentPrimaryHex:    strings.TrimSpace(out.AccentPrimaryHex),
		AccentSecondaryHex:  strings.TrimSpace(out.AccentSecondaryHex),
	}
	if len(resp.MainColorsHex) == 0 {
		resp.MainColorsHex = extractedColors // подстрахуемся палитрой из HTML
	}
	if resp.Summary == "" {
		resp.Summary = "Описание сайта недоступно."
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
	apiLogFinish(ctx, apiID, http.StatusOK, rawJSON, "", time.Since(apiStart))
}
