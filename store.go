package main

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"
)

type Store interface {
	CreateAPILog(ctx context.Context, route, urlStr, reqBody string) (APILogID, error)
	UpdateAPILog(ctx context.Context, id APILogID, status int, respBody, errText string, dur time.Duration) error
	CreateAILog(ctx context.Context, apiID *APILogID, model, promptPreview string) (AILogID, error)
	UpdateAILog(ctx context.Context, id AILogID, respBody, errText string, usage *CompletionUsageLike, dur time.Duration) error

	GetPrompt(key string, locale string, version int) (*Prompt, error)
	GetAllPrompts() ([]Prompt, error)
	GetPromptsByKeysLatestLocale(keys []string, locale string) ([]Prompt, error)
	UpdatePrompt(ctx context.Context, req PromptUpdateRequest) error

	GetRecentAPILogs(ctx context.Context) ([]APILogRow, error)
	GetRecentAILogs(ctx context.Context) ([]AILogRow, error)
}

type SQLStore struct {
	DB *sql.DB
}

func NewSQLStore(db *sql.DB) *SQLStore {
	return &SQLStore{DB: db}
}

func (s *SQLStore) CreateAPILog(ctx context.Context, route, urlStr, reqBody string) (APILogID, error) {
	res, err := s.DB.ExecContext(ctx,
		`INSERT INTO api_logs (route, url, request_body) VALUES (?,?,?)`,
		route, urlStr, reqBody,
	)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return APILogID(id), nil
}

func (s *SQLStore) UpdateAPILog(ctx context.Context, id APILogID, status int, respBody, errText string, dur time.Duration) error {
	if id == 0 {
		return nil
	}
	_, err := s.DB.ExecContext(ctx,
		`UPDATE api_logs 
           SET response_body=?, status_code=?, error_text=?, duration_ms=? 
         WHERE id=?`,
		respBody, status, nullIfEmpty(errText), int(dur.Milliseconds()), id,
	)
	return err
}

func (s *SQLStore) CreateAILog(ctx context.Context, apiID *APILogID, model, promptPreview string) (AILogID, error) {
	var apiRef interface{} = nil
	if apiID != nil && *apiID > 0 {
		apiRef = int64(*apiID)
	}
	res, err := s.DB.ExecContext(ctx,
		`INSERT INTO ai_logs (api_log_id, model, prompt_preview) VALUES (?,?,?)`,
		apiRef, model, promptPreview,
	)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return AILogID(id), nil
}

func (s *SQLStore) UpdateAILog(ctx context.Context, id AILogID, respBody, errText string, usage *CompletionUsageLike, dur time.Duration) error {
	if id == 0 {
		return nil
	}
	var pt, ct, tt interface{} = nil, nil, nil
	if usage != nil {
		pt = usage.PromptTokens
		ct = usage.CompletionTokens
		tt = usage.TotalTokens
	}
	_, err := s.DB.ExecContext(ctx,
		`UPDATE ai_logs 
           SET response_body=?, error_text=?, prompt_tokens=?, completion_tokens=?, total_tokens=?, duration_ms=?
         WHERE id=?`,
		respBody, nullIfEmpty(errText), pt, ct, tt, int(dur.Milliseconds()), id,
	)
	return err
}

func (s *SQLStore) GetPrompt(key string, locale string, version int) (*Prompt, error) {
	q := `
        SELECT id, key_name, locale, version, description, text, is_active, updated_by, updated_at
        FROM prompts
        WHERE key_name = ? AND locale = ?
          AND (version = ? OR ? = 0)
          AND is_active = 1
        ORDER BY version DESC
        LIMIT 1`
	row := s.DB.QueryRow(q, key, locale, version, version)

	var p Prompt
	if err := row.Scan(
		&p.ID, &p.KeyName, &p.Locale, &p.Version, &p.Description,
		&p.Text, &p.IsActive, &p.UpdatedBy, &p.UpdatedAt,
	); err != nil {
		return nil, err
	}
	return &p, nil
}

func (s *SQLStore) GetAllPrompts() ([]Prompt, error) {
	q := `
        SELECT id, key_name, locale, version, description, text, is_active, updated_by, updated_at
        FROM prompts
        WHERE is_active = 1
        ORDER BY key_name, locale, version DESC`
	rows, err := s.DB.Query(q)
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

func (s *SQLStore) GetPromptsByKeysLatestLocale(keys []string, locale string) ([]Prompt, error) {
	if len(keys) == 0 {
		return []Prompt{}, nil
	}

	placeholders := strings.TrimRight(strings.Repeat("?,", len(keys)), ",")

	q := fmt.Sprintf(`
        SELECT p.id, p.key_name, p.locale, p.version, p.description, p.text, p.is_active, p.updated_by, p.updated_at
        FROM prompts p
        JOIN (
          SELECT key_name, MAX(version) AS version
          FROM prompts
          WHERE is_active = 1 AND locale = ? AND key_name IN (%s)
          GROUP BY key_name
        ) t ON t.key_name = p.key_name AND t.version = p.version
        WHERE p.is_active = 1 AND p.locale = ?
        ORDER BY p.key_name
    `, placeholders)

	args := make([]interface{}, 0, len(keys)+2)
	args = append(args, locale)
	for _, k := range keys {
		args = append(args, k)
	}
	args = append(args, locale)

	rows, err := s.DB.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Prompt
	for rows.Next() {
		var p Prompt
		if err := rows.Scan(
			&p.ID, &p.KeyName, &p.Locale, &p.Version, &p.Description,
			&p.Text, &p.IsActive, &p.UpdatedBy, &p.UpdatedAt,
		); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *SQLStore) UpdatePrompt(ctx context.Context, req PromptUpdateRequest) error {
	if strings.TrimSpace(req.KeyName) == "" || strings.TrimSpace(req.Locale) == "" || req.Version <= 0 {
		return fmt.Errorf("invalid prompt identifiers")
	}
	if strings.TrimSpace(req.Text) == "" {
		return fmt.Errorf("prompt text cannot be empty")
	}

	setClauses := []string{"text=?"}
	params := []interface{}{req.Text}

	if req.Description != nil {
		setClauses = append(setClauses, "description=?")
		params = append(params, strings.TrimSpace(*req.Description))
	}
	if req.IsActive != nil {
		setClauses = append(setClauses, "is_active=?")
		params = append(params, *req.IsActive)
	}

	updatedBy := strings.TrimSpace(req.UpdatedBy)
	if updatedBy == "" {
		updatedBy = "system"
	}
	setClauses = append(setClauses, "updated_by=?")
	params = append(params, updatedBy)

	query := fmt.Sprintf("UPDATE prompts SET %s WHERE key_name=? AND locale=? AND version=?", strings.Join(setClauses, ", "))
	params = append(params, req.KeyName, req.Locale, req.Version)

	res, err := s.DB.ExecContext(ctx, query, params...)
	if err != nil {
		return err
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if affected == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *SQLStore) GetRecentAPILogs(ctx context.Context) ([]APILogRow, error) {
	rows, err := s.DB.QueryContext(ctx, `
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

func (s *SQLStore) GetRecentAILogs(ctx context.Context) ([]AILogRow, error) {
	rows, err := s.DB.QueryContext(ctx, `
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
