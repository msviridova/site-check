// prompt_handler.go
package main

import (
	"encoding/json"
	"net/http"
	"strings"
)

// promptHandler обрабатывает POST /prompts - получение промпта по ключу
func promptHandler(w http.ResponseWriter, r *http.Request) {
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
	prompt, err := getPrompt(db, req.KeyName, req.Locale, req.Version)
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
func promptListHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "use GET", http.StatusMethodNotAllowed)
		return
	}

	// Получение всех промптов из БД
	prompts, err := getAllPrompts(db)
	if err != nil {
		http.Error(w, "database error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(prompts)
}
