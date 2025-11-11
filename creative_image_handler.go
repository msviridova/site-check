package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/openai/openai-go/v2"
)

// ---------- DTO ----------

type ImageConcept map[string]any

type imageResponse struct {
	URL  string `json:"url"`            // /static/…png
	Size string `json:"size,omitempty"` // исходный (или нормализованный) размер
	// можно добавить другие поля по желанию
}

// ---------- JSON helpers ----------

type jsonErr struct {
	Error  string `json:"error"`
	Status int    `json:"status"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, jsonErr{Error: msg, Status: status})
}

// ---------- handler ----------

func (app *App) imageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSONError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	if ct := r.Header.Get("Content-Type"); ct != "" && !strings.Contains(ct, "application/json") {
		writeJSONError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return
	}

	var in ImageTemplate
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeJSONError(w, http.StatusBadRequest, "bad JSON: "+err.Error())
		return
	}
	_ = r.Body.Close()

	ctx, cancel := context.WithTimeout(r.Context(), app.Config.AITimeout)
	defer cancel()

	rawReq, _ := json.Marshal(in)
	entry := app.beginAPILog(ctx, "/image", "", string(rawReq))

	result, errObj := app.executeImage(ctx, in)
	if errObj != nil {
		logRouteError("/image", errObj)
		writeJSONError(w, errObj.Status, errObj.ClientMessage)
		entry.Finish(errObj.Status, errObj.RawBody, errObj.LogMessage)
		return
	}

	writeJSON(w, http.StatusOK, result.Response)
	entry.Finish(http.StatusOK, result.RawAI, "")
}

type imageResult struct {
	Response imageResponse
	RawAI    string
}

type imagePromptPlan struct {
	Prompt    string `json:"prompt"`
	Negatives string `json:"negatives"`
}

func (app *App) executeImage(ctx context.Context, in ImageTemplate) (*imageResult, *httpError) {
	conceptStr := strings.TrimSpace(string(in.Concept))
	if conceptStr == "" {
		return nil, newHTTPError(http.StatusBadRequest, "field 'concept' is required", "concept missing", nil)
	}
	if !json.Valid(in.Concept) {
		return nil, newHTTPError(http.StatusBadRequest, "field 'concept' must be valid JSON", "concept invalid json", nil)
	}
	if limit := app.Config.ImageConceptMaxBytes; limit > 0 && len(conceptStr) > limit {
		return nil, newHTTPError(http.StatusBadRequest, "field 'concept' is too long", "concept too long", nil)
	}

	additional := strings.TrimSpace(in.Additional)
	if limit := app.Config.ImageAdditionalMaxBytes; limit > 0 && len(additional) > limit {
		return nil, newHTTPError(http.StatusBadRequest, "field 'additional' is too long", "additional too long", nil)
	}
	if additional == "" {
		additional = DefaultImageAdditional
	}

	if app.Store == nil {
		return nil, newHTTPError(http.StatusInternalServerError, "store not initialized", "store not initialized", nil)
	}

	promptTemplate, err := app.getPromptCached("image", "ru", 0)
	if err != nil {
		return nil, newHTTPError(http.StatusInternalServerError, "prompt not found: image", "no prompt in DB", err)
	}

	payload := applyPlaceholders(promptTemplate.Text, map[string]string{
		"{CONCEPT}":    conceptStr,
		"{ADDITIONAL}": additional,
	})

	logInfo("image payload", map[string]interface{}{"payload": payload})

	raw, _, err := app.callChatCompletion(ctx, payload, AIRequestOptions{
		MaxTokens:   openai.Int(600),
		Temperature: openai.Float(0.2),
	})
	if err != nil {
		errMsg := err.Error()
		if errMsg == "no choices from AI" {
			return nil, newHTTPError(http.StatusBadGateway, "AI: no choices", "AI: no choices", err)
		}
		return nil, newHTTPError(http.StatusBadGateway, "AI error: "+errMsg, "ai error: "+errMsg, err)
	}

	var plan imagePromptPlan
	if err := json.Unmarshal([]byte(raw), &plan); err != nil {
		trimmed := extractJSONBlock(raw)
		if trimmed == "" || json.Unmarshal([]byte(trimmed), &plan) != nil {
			logWarn("image ai json parse failed", map[string]interface{}{"raw": raw, "error": err.Error()})
			httpErr := newHTTPError(http.StatusBadGateway, "AI JSON parse error: "+err.Error(), "ai json parse error", err)
			httpErr.RawBody = raw
			return nil, httpErr
		}
		raw = trimmed
	}

	finalPrompt := strings.TrimSpace(plan.Prompt)
	if finalPrompt == "" {
		return nil, newHTTPError(http.StatusBadGateway, "AI prompt is empty", "empty ai prompt", nil)
	}
	if plan.Negatives != "" {
		finalPrompt += ". Negative prompts: " + plan.Negatives
	}
	if additional != "" {
		finalPrompt += ". Restrictions: " + additional
	}

	size := strings.TrimSpace(in.Size)

	resultURL, err := app.generateImage(ctx, finalPrompt, size, "url")
	if err != nil {
		return nil, newHTTPError(http.StatusBadGateway, "AI error: "+err.Error(), "image generation failed", err)
	}

	return &imageResult{
		Response: imageResponse{URL: resultURL, Size: size},
		RawAI:    raw,
	}, nil
}

func extractJSONBlock(raw string) string {
	start := strings.Index(raw, "{")
	if start == -1 {
		return ""
	}
	// искать закрывающую скобку, учитывая вложенность
	depth := 0
	for i := start; i < len(raw); i++ {
		switch raw[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return raw[start : i+1]
			}
		}
	}
	return ""
}
