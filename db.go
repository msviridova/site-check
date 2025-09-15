// db.go
package main

import (
	"context"
	"database/sql"
	"log"
	"os"
	"strings"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

var db *sql.DB

func mustOpenDB() *sql.DB {
	dsn := os.Getenv("DATABASE_URL")
	if strings.TrimSpace(dsn) == "" {
		log.Fatal("DATABASE_URL is empty")
	}
	d, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatal(err)
	}
	d.SetMaxOpenConns(10)
	d.SetMaxIdleConns(10)
	d.SetConnMaxLifetime(30 * time.Minute)
	if err := d.Ping(); err != nil {
		log.Fatal(err)
	}
	return d
}

// ====== IDs для логов ======

type APILogID int64
type AILogID int64

// ====== API-логи ======

func apiLogStart(ctx context.Context, route, urlStr, reqBody string) (APILogID, error) {
	res, err := db.ExecContext(ctx,
		`INSERT INTO api_logs (route, url, request_body) VALUES (?,?,?)`,
		route, urlStr, reqBody,
	)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return APILogID(id), nil
}

func apiLogFinish(ctx context.Context, id APILogID, status int, respBody string, errText string, dur time.Duration) {
	_, err := db.ExecContext(ctx,
		`UPDATE api_logs 
		   SET response_body=?, status_code=?, error_text=?, duration_ms=? 
		 WHERE id=?`,
		respBody, status, nullIfEmpty(errText), int(dur.Milliseconds()), id,
	)
	if err != nil {
		log.Println("apiLogFinish error:", err)
	}
}

// ====== AI-логи ======

func aiLogStart(ctx context.Context, apiID *APILogID, model, promptPreview string) (AILogID, error) {
	var apiRef interface{} = nil
	if apiID != nil && *apiID > 0 {
		apiRef = int64(*apiID)
	}
	res, err := db.ExecContext(ctx,
		`INSERT INTO ai_logs (api_log_id, model, prompt_preview) VALUES (?,?,?)`,
		apiRef, model, promptPreview,
	)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return AILogID(id), nil
}

// CompletionUsageLike определяется в ai.go
func aiLogFinish(ctx context.Context, id AILogID, respBody, errText string, usage *CompletionUsageLike, dur time.Duration) {
	var pt, ct, tt interface{} = nil, nil, nil
	if usage != nil {
		pt = usage.PromptTokens
		ct = usage.CompletionTokens
		tt = usage.TotalTokens
	}
	_, err := db.ExecContext(ctx,
		`UPDATE ai_logs 
		   SET response_body=?, error_text=?, prompt_tokens=?, completion_tokens=?, total_tokens=?, duration_ms=?
		 WHERE id=?`,
		respBody, nullIfEmpty(errText), pt, ct, tt, int(dur.Milliseconds()), id,
	)
	if err != nil {
		log.Println("aiLogFinish error:", err)
	}
}

// ====== Вспомогательные ======

func nullIfEmpty(s string) interface{} {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}

// ====== PROMPTS ======
//
// Тип Prompt объявлен в types.go

func getPrompt(db *sql.DB, key string, locale string, version int) (*Prompt, error) {
	q := `
        SELECT id, key_name, locale, version, description, text, is_active, updated_by, updated_at
        FROM prompts
        WHERE key_name = ? AND locale = ?
          AND (version = ? OR ? = 0)
          AND is_active = 1
        ORDER BY version DESC
        LIMIT 1`
	row := db.QueryRow(q, key, locale, version, version)

	var p Prompt
	if err := row.Scan(
		&p.ID, &p.KeyName, &p.Locale, &p.Version, &p.Description,
		&p.Text, &p.IsActive, &p.UpdatedBy, &p.UpdatedAt,
	); err != nil {
		return nil, err
	}
	return &p, nil
}

func getAllPrompts(db *sql.DB) ([]Prompt, error) {
	q := `
        SELECT id, key_name, locale, version, description, text, is_active, updated_by, updated_at
        FROM prompts
        WHERE is_active = 1
        ORDER BY key_name, locale, version DESC`
	rows, err := db.Query(q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var prompts []Prompt
	for rows.Next() {
		var p Prompt
		if err := rows.Scan(
			&p.ID, &p.KeyName, &p.Locale, &p.Version, &p.Description,
			&p.Text, &p.IsActive, &p.UpdatedBy, &p.UpdatedAt,
		); err != nil {
			return nil, err
		}
		prompts = append(prompts, p)
	}
	return prompts, rows.Err()
}

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

func getRecentAPILogs(ctx context.Context) ([]APILogRow, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, route, url, status_code, error_text, duration_ms, created_at
		FROM api_logs
		ORDER BY id DESC
		LIMIT 50`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []APILogRow
	for rows.Next() {
		var r APILogRow
		if err := rows.Scan(&r.ID, &r.Route, &r.URL, &r.StatusCode, &r.ErrorText, &r.DurationMS, &r.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func getRecentAILogs(ctx context.Context) ([]AILogRow, error) {
	rows, err := db.QueryContext(ctx, `
		SELECT id, model, prompt_preview, error_text, total_tokens, created_at
		FROM ai_logs
		ORDER BY id DESC
		LIMIT 50`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []AILogRow
	for rows.Next() {
		var r AILogRow
		if err := rows.Scan(&r.ID, &r.Model, &r.PromptPreview, &r.ErrorText, &r.TotalTokens, &r.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
