// creative_handler.go
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

func creativeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	var req CreativeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}

	// ОБЯЗАТЕЛЕН уже извлечённый текст сайта (мы сайт НЕ скачиваем)
	siteText := strings.TrimSpace(req.SiteText)
	if siteText == "" {
		http.Error(w, "site_text is required", http.StatusBadRequest)
		return
	}

	// общий таймаут на обращение к AI
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	var resp CreativeResponse
	resp.Kind = req.Kind
	resp.Lang = "ru"
	resp.Source = "ai"

	// если глобально выключен AI — сразу 503
	if !useAI {
		resp.Source = "ai_error"
		http.Error(w, "AI disabled", http.StatusServiceUnavailable)
		return
	}

	switch strings.ToLower(req.Kind) {

	case "text":
		switch strings.ToLower(req.TextType) {
		case "keywords":
			kws, err := generateKeywords(ctx, siteText)
			if err != nil {
				resp.Source = "ai_error"
				http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
				return
			}
			resp.TextType = "keywords"
			resp.Keywords = kws

		case "negatives":
			negs, err := generateNegatives(ctx, siteText)
			if err != nil {
				resp.Source = "ai_error"
				http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
				return
			}
			resp.TextType = "negatives"
			resp.Negatives = negs

		case "ads":
			ads, err := generateAds(ctx, siteText)
			if err != nil {
				resp.Source = "ai_error"
				http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
				return
			}
			resp.TextType = "ads"
			resp.Ads = ads

		default:
			http.Error(w, "text_type must be: keywords | negatives | ads", http.StatusBadRequest)
			return
		}

	case "graphic":
		opts := GraphicInputOpts{
			Goal:             req.Goal,
			Audience:         req.Audience,
			Geo:              req.Geo,
			OfferConstraints: req.OfferConstraints,
			BrandOverrides:   req.BrandOverrides,
		}
		gp, err := generateGraphic(ctx, strings.TrimSpace(req.SiteURL), siteText, opts)
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
