package main

import (
	"context"
	"encoding/json"
	"log"
	"time"
)

type APILogEntry struct {
	app   *App
	ctx   context.Context
	id    APILogID
	start time.Time
	route string
}

func logFields(level, message string, fields map[string]interface{}) {
	entry := map[string]interface{}{
		"level": level,
		"msg":   message,
		"time":  time.Now().UTC().Format(time.RFC3339Nano),
	}
	for k, v := range fields {
		entry[k] = v
	}
	data, err := json.Marshal(entry)
	if err != nil {
		// Fallback на стандартный лог
		log.Printf("%s: %s %+v", level, message, fields)
		return
	}
	log.Println(string(data))
}

func logInfo(message string, fields map[string]interface{}) {
	logFields("INFO", message, fields)
}

func logWarn(message string, fields map[string]interface{}) {
	logFields("WARN", message, fields)
}

func logError(message string, fields map[string]interface{}) {
	logFields("ERROR", message, fields)
}

func (app *App) beginAPILog(ctx context.Context, route, urlStr, reqBody string) *APILogEntry {
	entry := &APILogEntry{app: app, ctx: ctx, start: time.Now(), route: route}
	if app == nil || app.Store == nil {
		logWarn("store not initialized; API logging disabled", map[string]interface{}{"route": route})
		return entry
	}
	id, err := app.Store.CreateAPILog(ctx, route, urlStr, reqBody)
	if err != nil {
		logWarn("apiLogStart failed", map[string]interface{}{"route": route, "error": err.Error()})
		return entry
	}
	entry.id = id
	return entry
}

func (e *APILogEntry) Finish(status int, respBody, errText string) {
	if e == nil || e.app == nil || e.app.Store == nil || e.id == 0 {
		return
	}
	if err := e.app.Store.UpdateAPILog(e.ctx, e.id, status, respBody, errText, time.Since(e.start)); err != nil {
		logWarn("apiLogFinish failed", map[string]interface{}{"route": e.route, "error": err.Error()})
	}
}
