// creative_handler.go
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
)

func (app *App) creativeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	var req CreativeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	_ = r.Body.Close()

	ctx, cancel := context.WithTimeout(r.Context(), app.Config.CreativeTimeout)
	defer cancel()

	rawReq, _ := json.Marshal(req)
	entry := app.beginAPILog(ctx, "/creative", strings.TrimSpace(req.SiteURL), string(rawReq))

	result, errObj := app.executeCreative(ctx, req)
	if errObj != nil {
		logRouteError("/creative", errObj)
		http.Error(w, errObj.ClientResponse(), errObj.Status)
		entry.Finish(errObj.Status, errObj.RawBody, errObj.LogMessage)
		return
	}

	rawResp, err := json.Marshal(result)
	if err != nil {
		logError("creative encode response failed", map[string]interface{}{"error": err.Error()})
		http.Error(w, "encode error", http.StatusInternalServerError)
		entry.Finish(http.StatusInternalServerError, "", "encode error")
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_, _ = w.Write(rawResp)
	entry.Finish(http.StatusOK, string(rawResp), "")
}

func (app *App) executeCreative(ctx context.Context, req CreativeRequest) (*CreativeResponse, *httpError) {
	kindOriginal := strings.TrimSpace(req.Kind)
	if kindOriginal == "" {
		kindOriginal = "graphic"
	}
	kind := strings.ToLower(kindOriginal)

	resp := &CreativeResponse{
		Kind:   kindOriginal,
		Lang:   "ru",
		Source: "ai",
	}

	siteText := sanitizeSiteText(req.SiteText, app.Config.CreativeSiteTextMax)
	siteURL := strings.TrimSpace(req.SiteURL)
	if siteText == "" && siteURL != "" {
		html, err := fetchHTML(ctx, siteURL, app.Config.HTTPClient)
		if err != nil {
			msg := "fetch error: " + err.Error()
			return nil, newHTTPError(http.StatusBadGateway, msg, msg, err)
		}
		siteText = sanitizeSiteText(extractVisibleText(html), app.Config.CreativeSiteTextMax)
	}

	if siteText == "" {
		return nil, newHTTPError(http.StatusBadRequest, "site_text is required", "site_text is required", nil)
	}

	switch kind {
	case "text":
		textCreatives, err := app.generateAllTextCreatives(ctx, siteText)
		if err != nil {
			msg := "ai error: " + err.Error()
			return nil, newHTTPError(http.StatusBadGateway, "AI error: "+err.Error(), msg, err)
		}
		resp.Keywords = textCreatives.Keywords
		resp.Negatives = textCreatives.Negatives
		resp.Ads = textCreatives.Ads

	case "graphic":
		opts := GraphicInputOpts{
			Goal:             strings.TrimSpace(req.Goal),
			Audience:         strings.TrimSpace(req.Audience),
			Geo:              strings.TrimSpace(req.Geo),
			OfferConstraints: strings.TrimSpace(req.OfferConstraints),
			BrandOverrides:   strings.TrimSpace(req.BrandOverrides),
		}
		gp, err := app.generateGraphic(ctx, siteURL, siteText, opts)
		if err != nil {
			msg := "ai error: " + err.Error()
			return nil, newHTTPError(http.StatusBadGateway, "AI error: "+err.Error(), msg, err)
		}
		resp.Graphic = gp

	default:
		return nil, newHTTPError(http.StatusBadRequest, "kind must be: text | graphic", "bad kind", nil)
	}

	return resp, nil
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
