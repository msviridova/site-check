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

type APILogID int64
type AILogID int64

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

func aiLogFinish(ctx context.Context, id AILogID, respBody, errText string, usage *CompletionUsageLike, dur time.Duration) {
	// usage совместимая «прокладка», см. ниже в ai.go
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

func nullIfEmpty(s string) interface{} {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}

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
