// creative_handler.go
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// локальный ответ (чтобы не лезть в types.go)
type creativeResponse struct {
	Kind    string       `json:"kind"`
	Lang    string       `json:"lang"`
	Source  string       `json:"source"`
	Graphic *GraphicPlan `json:"graphic,omitempty"`

	// Для текстовых креативов (все типы сразу)
	Keywords  []string  `json:"keywords,omitempty"`
	Negatives []string  `json:"negatives,omitempty"`
	Ads       []AdBlock `json:"ads,omitempty"`
}

func creativeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	// читаем JSON
	var req CreativeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	_ = r.Body.Close()

	// значения по умолчанию
	req.Kind = strings.TrimSpace(req.Kind)
	if req.Kind == "" {
		req.Kind = "graphic"
	}
	lang := "ru"

	// общий таймаут на работу с AI
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	// готовим ответ-заготовку
	resp := creativeResponse{
		Kind:   req.Kind,
		Lang:   lang,
		Source: "ai",
	}

	// если глобально выключен AI — сразу 503 (поведение как в твоей версии)
	if !useAI {
		resp.Source = "ai_error"
		http.Error(w, "AI disabled", http.StatusServiceUnavailable)
		return
	}

	// соберём siteText: приоритет у site_text; если его нет — попробуем подтянуть по URL
	siteText := sanitizeSiteText(req.SiteText, 10000)
	if siteText == "" && strings.TrimSpace(req.SiteURL) != "" {
		// тянем HTML и извлекаем видимый текст штатными функциями fetch.go
		html, err := fetchHTML(ctx, req.SiteURL)
		if err != nil {
			http.Error(w, "fetch error: "+err.Error(), http.StatusBadGateway)
			return
		}
		siteText = extractVisibleText(html)
		siteText = sanitizeSiteText(siteText, 10000)
	}

	if siteText == "" {
		http.Error(w, "site_text is required", http.StatusBadRequest)
		return
	}

	switch strings.ToLower(req.Kind) {

	case "text":
		// генерируем все типы текстовых креативов сразу
		textCreatives, err := generateAllTextCreatives(ctx, siteText)
		if err != nil {
			resp.Source = "ai_error"
			http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
			return
		}
		resp.Keywords = textCreatives.Keywords
		resp.Negatives = textCreatives.Negatives
		resp.Ads = textCreatives.Ads

	case "graphic":
		// собираем опции для графики из тела запроса
		opts := GraphicInputOpts{
			Goal:             req.Goal,
			Audience:         req.Audience,
			Geo:              req.Geo,
			OfferConstraints: req.OfferConstraints,
			BrandOverrides:   req.BrandOverrides,
		}

		gp, err := generateGraphic(ctx, req.SiteURL, siteText, opts)
		if err != nil {
			resp.Source = "ai_error"
			http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
			return
		}
		resp.Graphic = gp

	default:
		http.Error(w, "kind must be: text | graphic", http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}

// --- утилита очистки текста (мягкая версия) ---
func sanitizeSiteText(s string, max int) string {
	s = strings.ReplaceAll(s, "\uFEFF", "")
	s = strings.ReplaceAll(s, "\u00A0", " ")
	s = strings.TrimSpace(s)
	// уберём возможные кодовые блоки и хвосты
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	s = strings.TrimSuffix(s, "…")
	// схлопнём пробелы
	s = strings.Join(strings.Fields(s), " ")
	if max > 0 && len(s) > max {
		return s[:max]
	}
	return s
}
