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

// ==== HTTP-обработчик ====

func classifyHandler(w http.ResponseWriter, r *http.Request) {
	// 1) только POST
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	// 2) читаем JSON
	var req classifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}

	// 3) валидируем URL
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

	log.Printf("useAI=%v url=%s", useAI, u.String())

	// 4) общий таймаут
	ctx, cancel := context.WithTimeout(r.Context(), 12*time.Second)
	defer cancel()

	// старт лог-записи API
	rawReq, _ := json.Marshal(req)
	apiStart := time.Now()
	apiID, _ := apiLogStart(ctx, "/classify", u.String(), string(rawReq))

	// 5) скачиваем HTML
	html, err := fetchHTML(ctx, u.String())
	if err != nil {
		http.Error(w, "fetch failed: "+err.Error(), http.StatusBadGateway)
		apiLogFinish(ctx, apiID, http.StatusBadGateway, "", "fetch failed: "+err.Error(), time.Since(apiStart))
		return
	}

	// --- бренд/цвета/стиль (делаем один раз)
	brand := extractBrand(u, html)
	palette := extractColorsHex(html)
	styleNotes := deriveStyleNotes(palette, html)

	// 6) извлекаем текст
	text := extractVisibleText(html)
	log.Printf("extracted text length: %d", len(text))

	// === мало текста: пробуем AI по домену/титлу, иначе фолбэк
	if len(strings.TrimSpace(text)) < 40 {
		brief := fallbackSummary(u, html) // title/meta/host
		if useAI {
			shortInput := "Домен: " + u.Hostname()
			if b := strings.TrimSpace(brief); b != "" {
				shortInput += "\nTitle/Meta: " + b
			}
			sum, kws, negs, cols, aiErr := summarizeWithAI(ctx, shortInput)
			log.Printf("AI (short-text) finished, err=%v", aiErr)
			if aiErr == nil && strings.TrimSpace(sum) != "" {
				resp := classifyResponse{
					Summary:  sum,
					Lang:     "ru",
					Source:   "ai",
					Keywords: kws, NegativeKeywords: negs,

					Brand:      brand,
					StyleNotes: styleNotes,

					MainColorsHex:       nilIfNilSlice(cols, func(c *AIColors) []string { return c.Main }),
					AdditionalColorsHex: nilIfNilSlice(cols, func(c *AIColors) []string { return c.Additional }),
					BackgroundColorHex:  ifCols(cols, func(c *AIColors) string { return c.Background }),
					AccentPrimaryHex:    ifCols(cols, func(c *AIColors) string { return c.AccentPrimary }),
					AccentSecondaryHex:  ifCols(cols, func(c *AIColors) string { return c.AccentSecondary }),
				}
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				_ = json.NewEncoder(w).Encode(resp)

				respJSON, _ := json.Marshal(resp)
				apiLogFinish(ctx, apiID, http.StatusOK, string(respJSON), "", time.Since(apiStart))
				return
			}
			log.Println("AI short-text failed → fallback to heuristic")
		}

		// эвристический фолбэк
		summary := brief
		if strings.TrimSpace(summary) == "" {
			summary = "Веб-сайт компании/сервиса " + u.Hostname()
		}
		resp := classifyResponse{
			Summary:    summary,
			Lang:       "ru",
			Source:     "heuristic",
			Brand:      brand,
			StyleNotes: styleNotes,
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(resp)

		respJSON, _ := json.Marshal(resp)
		apiLogFinish(ctx, apiID, http.StatusOK, string(respJSON), "", time.Since(apiStart))
		return
	}

	// === текста достаточно: обычная логика
	var (
		summary string
		source  string
		kws     []string
		negs    []string
		cols    *AIColors
	)
	if useAI {
		source = "ai"
		sum, kk, nn, cc, aiErr := summarizeWithAI(ctx, text)
		log.Printf("AI call finished, err=%v", aiErr)
		if aiErr != nil || strings.TrimSpace(sum) == "" {
			log.Println("AI failed or empty → fallback to heuristic")
			summary = heuristicSummarize(text)
			source = "heuristic"
		} else {
			summary, kws, negs, cols = sum, kk, nn, cc
		}
	} else {
		summary = heuristicSummarize(text)
		source = "heuristic"
	}

	// стоп-фолбэк
	if strings.TrimSpace(summary) == "" {
		log.Println("summary is empty → using title/meta/host fallback")
		summary = fallbackSummary(u, html)
		if strings.TrimSpace(summary) == "" {
			summary = "Не удалось определить тематику сайта"
		}
	}

	// ответ + финал лога
	resp := classifyResponse{
		Summary:  summary,
		Lang:     "ru",
		Source:   source,
		Keywords: kws, NegativeKeywords: negs,

		Brand:      brand,
		StyleNotes: styleNotes,

		MainColorsHex:       nilIfNilSlice(cols, func(c *AIColors) []string { return c.Main }),
		AdditionalColorsHex: nilIfNilSlice(cols, func(c *AIColors) []string { return c.Additional }),
		BackgroundColorHex:  ifCols(cols, func(c *AIColors) string { return c.Background }),
		AccentPrimaryHex:    ifCols(cols, func(c *AIColors) string { return c.AccentPrimary }),
		AccentSecondaryHex:  ifCols(cols, func(c *AIColors) string { return c.AccentSecondary }),
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)

	respJSON, _ := json.Marshal(resp)
	apiLogFinish(ctx, apiID, http.StatusOK, string(respJSON), "", time.Since(apiStart))
}

// безопасные «мостики» чтобы не натыкаться на nil
func ifCols[T any](c *AIColors, get func(*AIColors) T) T {
	var zero T
	if c == nil {
		return zero
	}
	return get(c)
}

func nilIfNilSlice[T any](c *AIColors, get func(*AIColors) []T) []T {
	if c == nil {
		return nil
	}
	v := get(c)
	if len(v) == 0 {
		return nil
	}
	return v
}
