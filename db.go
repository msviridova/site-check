// db.go
package main

import (
	"strings"
	"time"
)

// ====== IDs для логов ======

type APILogID int64
type AILogID int64

// ====== Вспомогательные ======

func nullIfEmpty(s string) interface{} {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}

// ====== Структуры для логов ======

type APILogRow struct {
	ID         int64     `json:"id"`
	Route      string    `json:"route"`
	URL        string    `json:"url"`
	StatusCode *int      `json:"status_code,omitempty"`
	ErrorText  *string   `json:"error_text,omitempty"`
	DurationMS *int      `json:"duration_ms,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
}

type AILogRow struct {
	ID            int64     `json:"id"`
	Model         string    `json:"model"`
	PromptPreview string    `json:"prompt_preview"`
	ErrorText     *string   `json:"error_text,omitempty"`
	TotalTokens   *int      `json:"total_tokens,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
}
