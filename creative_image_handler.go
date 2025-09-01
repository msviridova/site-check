// creative_image_handler.go
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
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

	if strings.TrimSpace(req.Size) == "" {
		req.Size = "1:1"
	}
	if strings.TrimSpace(req.ResponseFormat) == "" {
		req.ResponseFormat = "url"
	}

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
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
			// сохраняем во временный PNG и открываем
			data, _ := base64.StdEncoding.DecodeString(out)
			tmp, _ := os.CreateTemp("", "img-*.png")
			defer tmp.Close()
			_ = os.WriteFile(tmp.Name(), data, 0644)
			_ = exec.Command("open", tmp.Name()).Start()
		} else {
			resp.URL = out
			if strings.HasPrefix(out, "data:image/png;base64,") {
				raw := strings.TrimPrefix(out, "data:image/png;base64,")
				data, _ := base64.StdEncoding.DecodeString(raw)
				tmp, _ := os.CreateTemp("", "img-*.png")
				defer tmp.Close()
				_ = os.WriteFile(tmp.Name(), data, 0644)
				_ = exec.Command("open", tmp.Name()).Start()
			}
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

	if req.Size == "" {
		req.Size = "1:1"
	}
	if req.ResponseFormat == "" {
		req.ResponseFormat = "url"
	}

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	out, err := generateImage(ctx, req.Prompt, req.Size, req.ResponseFormat)

	resp := ImageResponse{
		Prompt:         req.Prompt,
		Size:           req.Size,
		ResponseFormat: req.ResponseFormat,
		Lang:           "ru",
		Source:         "ai",
	}

	if err != nil {
		resp.Source = "ai_error"
		resp.Error = err.Error()
	} else if req.ResponseFormat == "b64_json" {
		resp.B64JSON = out
		data, _ := base64.StdEncoding.DecodeString(out)
		tmp, _ := os.CreateTemp("", "img-*.png")
		defer tmp.Close()
		_ = os.WriteFile(tmp.Name(), data, 0644)
		_ = exec.Command("open", tmp.Name()).Start()
	} else {
		resp.URL = out
		if strings.HasPrefix(out, "data:image/png;base64,") {
			raw := strings.TrimPrefix(out, "data:image/png;base64,")
			data, _ := base64.StdEncoding.DecodeString(raw)
			tmp, _ := os.CreateTemp("", "img-*.png")
			defer tmp.Close()
			_ = os.WriteFile(tmp.Name(), data, 0644)
			_ = exec.Command("open", tmp.Name()).Start()
		}
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}
