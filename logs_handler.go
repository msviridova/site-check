// logs_handler.go
package main

import (
	"encoding/json"
	"net/http"
)

func logsHandler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	api, _ := getRecentAPILogs(ctx)
	ai, _ := getRecentAILogs(ctx)

	// гарантируем пустые массивы (а не null)
	if api == nil {
		api = []APILogRow{}
	}
	if ai == nil {
		ai = []AILogRow{}
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(struct {
		API []APILogRow `json:"api"`
		AI  []AILogRow  `json:"ai"`
	}{
		API: api,
		AI:  ai,
	})
}
