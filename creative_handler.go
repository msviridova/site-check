// creative_handler.go
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/url"
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

	// общий таймаут (подлиннее, т.к. возможна генерация картинок)
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	// лог API
	rawReq, _ := json.Marshal(req)
	apiStart := time.Now()
	apiID, _ := apiLogStart(ctx, "/creative", u.String(), string(rawReq))

	// забираем контент
	html, err := fetchHTML(ctx, u.String())
	if err != nil {
		http.Error(w, "fetch failed: "+err.Error(), http.StatusBadGateway)
		apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "fetch failed: "+err.Error(), time.Since(apiStart))
		return
	}
	siteText := extractVisibleText(html)
	if len(siteText) > 12000 {
		siteText = siteText[:12000]
	}

	var resp CreativeResponse
	resp.Kind = req.Kind
	resp.Lang = "ru"
	resp.Source = "ai"

	if !useAI {
		resp.Source = "ai_error"
		http.Error(w, "AI disabled", http.StatusServiceUnavailable)
		apiLogFinish(ctx, apiID, http.StatusServiceUnavailable, "", "AI disabled", time.Since(apiStart))
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
				apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "AI error: "+err.Error(), time.Since(apiStart))
				return
			}
			resp.TextType = "keywords"
			resp.Keywords = kws

		case "negatives":
			negs, err := generateNegatives(ctx, siteText)
			if err != nil {
				resp.Source = "ai_error"
				http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
				apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "AI error: "+err.Error(), time.Since(apiStart))
				return
			}
			resp.TextType = "negatives"
			resp.Negatives = negs

		case "ads":
			ads, err := generateAds(ctx, siteText)
			if err != nil {
				resp.Source = "ai_error"
				http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
				apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "AI error: "+err.Error(), time.Since(apiStart))
				return
			}
			resp.TextType = "ads"
			resp.Ads = ads

		default:
			http.Error(w, "text_type must be: keywords | negatives | ads", http.StatusBadRequest)
			apiLogFinish(ctx, apiID, http.StatusBadRequest, "", "bad text_type", time.Since(apiStart))
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
		gp, err := generateGraphic(ctx, u.String(), siteText, opts)
		if err != nil {
			resp.Source = "ai_error"
			http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
			apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "AI error: "+err.Error(), time.Since(apiStart))
			return
		}

		// Генерация изображений сразу в base64 и вклейка в image_urls
		for i := range gp.Concepts {
			// 1x1
			if prompt := strings.TrimSpace(gp.Concepts[i].ImagePrompts.Sq1x1); prompt != "" {
				if b64, err := generateImage(ctx, prompt, "1024x1024", "b64_json"); err == nil {
					gp.Concepts[i].ImageURLs.Sq1x1 = b64
				} else {
					log.Printf("[image] 1x1 generation failed: %v", err)
				}
			}
			// 4x1 → используется допустимый размер 1536x1024 (широкий)
			if prompt := strings.TrimSpace(gp.Concepts[i].ImagePrompts.Ar4x1); prompt != "" {
				if b64, err := generateImage(ctx, prompt, "1536x1024", "b64_json"); err == nil {
					gp.Concepts[i].ImageURLs.Ar4x1 = b64
				} else {
					log.Printf("[image] 4x1 generation failed: %v", err)
				}
			}
			// 1x2 → используем 1024x1536 (вертикальный)
			if prompt := strings.TrimSpace(gp.Concepts[i].ImagePrompts.Ar1x2); prompt != "" {
				if b64, err := generateImage(ctx, prompt, "1024x1536", "b64_json"); err == nil {
					gp.Concepts[i].ImageURLs.Ar1x2 = b64
				} else {
					log.Printf("[image] 1x2 generation failed: %v", err)
				}
			}
		}

		resp.Graphic = gp

	default:
		http.Error(w, "kind must be: text | graphic", http.StatusBadRequest)
		apiLogFinish(ctx, apiID, http.StatusBadRequest, "", "bad kind", time.Since(apiStart))
		return
	}

	// успешный ответ
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	b, _ := json.Marshal(resp)
	_, _ = w.Write(b)
	apiLogFinish(ctx, apiID, http.StatusOK, string(b), "", time.Since(apiStart))
}
