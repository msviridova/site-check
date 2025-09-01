// creative_handler.go
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

// choosePromptAndAspectFromStrings выбирает промпт и размер по предпочтению.
// Если preferred пустой или неизвестный — берём 1x1.
func choosePromptAndAspectFromStrings(sq1x1, ar4x1, ar1x2, preferred string) (prompt string, aspect string, size string) {
	switch strings.TrimSpace(preferred) {
	case "4x1":
		if p := strings.TrimSpace(ar4x1); p != "" {
			return p, "4x1", "1536x1024"
		}
		// fallback к квадрату
		if p := strings.TrimSpace(sq1x1); p != "" {
			return p, "1x1", ""
		}
		if p := strings.TrimSpace(ar1x2); p != "" {
			return p, "1x2", "1024x1536"
		}
	case "1x2":
		if p := strings.TrimSpace(ar1x2); p != "" {
			return p, "1x2", "1024x1536"
		}
		// fallback к квадрату
		if p := strings.TrimSpace(sq1x1); p != "" {
			return p, "1x1", ""
		}
		if p := strings.TrimSpace(ar4x1); p != "" {
			return p, "4x1", "1536x1024"
		}
	default: // "1x1" или пусто
		if p := strings.TrimSpace(sq1x1); p != "" {
			// для квадрата размер не указываем — API возьмёт default 1024x1024
			return p, "1x1", ""
		}
		// fallback: возьмём что есть
		if p := strings.TrimSpace(ar4x1); p != "" {
			return p, "4x1", "1536x1024"
		}
		if p := strings.TrimSpace(ar1x2); p != "" {
			return p, "1x2", "1024x1536"
		}
	}
	// если вообще пусто — вернём пустые
	return "", "", ""
}

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

	siteText := strings.TrimSpace(req.SiteText)
	if siteText == "" {
		http.Error(w, "site_text is required", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 75*time.Second)
	defer cancel()

	// лог API
	rawReq, _ := json.Marshal(req)
	apiStart := time.Now()
	apiID, _ := apiLogStart(ctx, "/creative", strings.TrimSpace(req.SiteURL), string(rawReq))

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
		// Генерируем все типы текстовых креативов за один запрос
		textCreatives, err := generateAllTextCreatives(ctx, siteText)
		if err != nil {
			resp.Source = "ai_error"
			http.Error(w, "AI error: "+err.Error(), http.StatusBadGateway)
			apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "AI error: "+err.Error(), time.Since(apiStart))
			return
		}
		resp.Keywords = textCreatives.Keywords
		resp.Negatives = textCreatives.Negatives
		resp.Ads = textCreatives.Ads

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
			apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "AI error: "+err.Error(), time.Since(apiStart))
			return
		}
		resp.Graphic = gp

		// Сразу генерим одну картинку по предпочтению запроса (или 1x1 по умолчанию).
		if len(gp.Concepts) > 0 {
			c0 := gp.Concepts[0]
			prompt, aspect, size := choosePromptAndAspectFromStrings(
				c0.ImagePrompts.Sq1x1,
				c0.ImagePrompts.Ar4x1,
				c0.ImagePrompts.Ar1x2,
				req.PreferredAspect, // "1x1" | "4x1" | "1x2" | ""
			)
			if strings.TrimSpace(prompt) != "" {
				// берём base64 — чтобы сразу сохранить и открыть PNG
				b64, genErr := generateImage(ctx, prompt, size, "b64_json")
				if genErr == nil && strings.TrimSpace(b64) != "" {
					// сохраняем PNG
					ts := time.Now().Format("20060102_150405")
					if aspect == "" {
						aspect = "1x1"
					}
					filename := fmt.Sprintf("creative_%s_%s.png", ts, aspect)
					if data, decErr := base64.StdEncoding.DecodeString(b64); decErr == nil {
						_ = os.WriteFile(filename, data, 0644)
						// попробуем открыть (macOS); если не получится — просто игнорируем
						_ = exec.Command("open", filename).Start()
					}
				}
				// если не удалось — просто продолжим без падения; JSON-ответ вернём как есть (gp)
			}
		}

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
