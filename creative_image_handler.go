package main

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

func imageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}
	var req ImageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}

	// дефолты
	if strings.TrimSpace(req.Size) == "" {
		req.Size = "1024x1024"
	}
	if strings.TrimSpace(req.ResponseFormat) == "" {
		req.ResponseFormat = "url"
	}

	ctx, cancel := context.WithTimeout(r.Context(), 45*time.Second)
	defer cancel()

	out, err := generateImage(ctx, req.Prompt, req.Size, req.ResponseFormat)

	resp := ImageResponse{
		Prompt:         strings.TrimSpace(req.Prompt),
		Size:           req.Size,
		ResponseFormat: req.ResponseFormat,
		Lang:           "ru",
	}
	if err != nil {
		resp.Source = "ai_error"
		resp.Error = err.Error()
	} else {
		resp.Source = "ai"
		if req.ResponseFormat == "b64_json" {
			resp.B64JSON = out
		} else {
			resp.URL = out
		}
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}

func creativeImageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	var req ImageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}

	// дефолты
	if req.Size == "" {
		req.Size = "1024x1024"
	}
	if req.ResponseFormat == "" {
		req.ResponseFormat = "url"
	}

	ctx, cancel := context.WithTimeout(r.Context(), 45*time.Second)
	defer cancel()

	out, err := generateImage(ctx, req.Prompt, req.Size, req.ResponseFormat)
	if err != nil {
		resp := ImageResponse{
			Prompt:         req.Prompt,
			Size:           req.Size,
			ResponseFormat: req.ResponseFormat,
			Lang:           "ru",
			Source:         "ai_error",
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(resp)
		return
	}

	resp := ImageResponse{
		Prompt:         req.Prompt,
		Size:           req.Size,
		ResponseFormat: req.ResponseFormat,
		Lang:           "ru",
		Source:         "ai",
	}
	if req.ResponseFormat == "b64_json" {
		resp.B64JSON = out
	} else {
		resp.URL = out
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}
