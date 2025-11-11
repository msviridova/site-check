// prompt_handler.go
package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

// promptHandler обрабатывает POST /prompts - получение промпта по ключу
func (app *App) promptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	var req PromptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}

	// Валидация обязательных полей
	if strings.TrimSpace(req.KeyName) == "" {
		http.Error(w, "key_name is required", http.StatusBadRequest)
		return
	}

	// Значения по умолчанию
	if strings.TrimSpace(req.Locale) == "" {
		req.Locale = "ru"
	}
	if req.Version < 0 {
		req.Version = 0
	}

	// Получение промпта из БД
	if app.Store == nil {
		http.Error(w, "store not initialized", http.StatusInternalServerError)
		return
	}

	prompt, err := app.Store.GetPrompt(req.KeyName, req.Locale, req.Version)
	if err != nil {
		http.Error(w, "prompt not found", http.StatusNotFound)
		return
	}

	// Формирование ответа
	resp := PromptResponse{
		KeyName: prompt.KeyName,
		Locale:  prompt.Locale,
		Version: prompt.Version,
		Text:    prompt.Text,
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}

// promptListHandler обрабатывает GET /prompts/list - список всех промптов
func (app *App) promptListHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "use GET", http.StatusMethodNotAllowed)
		return
	}

	// Возвращаем ТОЛЬКО нужные 4 промпта для локали "ru"
	allowed := []string{"classify", "creative_text_all", "creative_graphic", "image"}
	if app.Store == nil {
		http.Error(w, "store not initialized", http.StatusInternalServerError)
		return
	}

	prompts, err := app.Store.GetPromptsByKeysLatestLocale(allowed, "ru")
	if err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(prompts)
}

func (app *App) promptUpdateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut && r.Method != http.MethodPost {
		http.Error(w, "use PUT", http.StatusMethodNotAllowed)
		return
	}

	var req PromptUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}

	if strings.TrimSpace(req.KeyName) == "" {
		http.Error(w, "key_name is required", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Locale) == "" {
		req.Locale = "ru"
	}
	if req.Version <= 0 {
		http.Error(w, "version must be > 0", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Text) == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}

	if app.Store == nil {
		http.Error(w, "store not initialized", http.StatusInternalServerError)
		return
	}

	if err := app.Store.UpdatePrompt(r.Context(), req); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.Error(w, "prompt not found", http.StatusNotFound)
			return
		}
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
