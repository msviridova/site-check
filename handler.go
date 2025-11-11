// handler.go
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"

	"github.com/openai/openai-go/v2"
)

func (app *App) classifyHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	var req classifyRequest
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

	ctx, cancel := context.WithTimeout(r.Context(), app.Config.ClassifyTimeout)
	defer cancel()

	rawReq, _ := json.Marshal(req)
	entry := app.beginAPILog(ctx, "/classify", u.String(), string(rawReq))

	result, errObj := app.executeClassify(ctx, u)
	if errObj != nil {
		logRouteError("/classify", errObj)
		http.Error(w, errObj.ClientResponse(), errObj.Status)
		entry.Finish(errObj.Status, errObj.RawBody, errObj.LogMessage)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if err := json.NewEncoder(w).Encode(result.Response); err != nil {
		logError("classify encode response failed", map[string]interface{}{"error": err.Error()})
		http.Error(w, "encode error", http.StatusInternalServerError)
		entry.Finish(http.StatusInternalServerError, "", "encode error")
		return
	}
	entry.Finish(http.StatusOK, result.RawAI, "")
}

type classifyResult struct {
	Response classifyResponse
	RawAI    string
}

func (app *App) executeClassify(ctx context.Context, u *url.URL) (*classifyResult, *httpError) {
	html, err := fetchHTML(ctx, u.String(), app.Config.HTTPClient)
	if err != nil {
		msg := "fetch failed: " + err.Error()
		return nil, newHTTPError(http.StatusBadGateway, msg, msg, err)
	}

	brandHeur := extractBrand(u, html)
	extractedColors := extractColorsHex(html)
	styleHeur := deriveStyleNotes(extractedColors, html)

	siteText := extractVisibleText(html)
	if limit := app.Config.ClassifySiteTextMax; limit > 0 && len(siteText) > limit {
		siteText = siteText[:limit]
	}
	siteText = strings.TrimSpace(siteText)

	if app.Store == nil {
		return nil, newHTTPError(http.StatusInternalServerError, "store not initialized", "store not initialized", nil)
	}

	const promptKey = "classify"
	p, err := app.Store.GetPrompt(promptKey, "ru", 0)
	if err != nil {
		return nil, newHTTPError(http.StatusInternalServerError, "prompt not found: "+promptKey, "no prompt in DB", err)
	}
	prompt := strings.TrimSpace(p.Text)
	if prompt == "" {
		return nil, newHTTPError(http.StatusInternalServerError, "prompt is empty: "+promptKey, "empty prompt text", nil)
	}

	colorsValue := "[]"
	if len(extractedColors) > 0 {
		colorsValue = strings.Join(extractedColors, ", ")
	}

	prompt = applyPlaceholders(prompt, map[string]string{
		"{HEUR_BRAND}":  brandHeur,
		"{HEUR_STYLE}":  styleHeur,
		"{HEUR_COLORS}": colorsValue,
		"{SITE_TEXT}":   siteText,
	})

	rawAI, _, err := app.callChatCompletion(ctx, prompt, AIRequestOptions{
		MaxTokens:   openai.Int(900),
		Temperature: openai.Float(0.3),
	})
	if err != nil {
		errMsg := err.Error()
		if errMsg == "no choices from AI" {
			return nil, newHTTPError(http.StatusBadGateway, "AI: no choices", "AI: no choices", err)
		}
		return nil, newHTTPError(http.StatusBadGateway, "AI error: "+errMsg, "ai error: "+errMsg, err)
	}

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
	if err := json.Unmarshal([]byte(rawAI), &out); err != nil {
		httpErr := newHTTPError(http.StatusBadGateway, "AI JSON parse error: "+err.Error(), "ai json parse error", err)
		httpErr.RawBody = rawAI
		return nil, httpErr
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
		MainColorsHex:       dedup(out.MainColorsHex),
		AdditionalColorsHex: dedup(out.AdditionalColorsHex),
		BackgroundColorHex:  strings.TrimSpace(out.BackgroundColorHex),
		AccentPrimaryHex:    strings.TrimSpace(out.AccentPrimaryHex),
		AccentSecondaryHex:  strings.TrimSpace(out.AccentSecondaryHex),
	}
	if len(resp.MainColorsHex) == 0 {
		resp.MainColorsHex = extractedColors
	}
	if resp.Summary == "" {
		resp.Summary = "Описание сайта недоступно."
	}

	return &classifyResult{Response: resp, RawAI: rawAI}, nil
}
