package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"golang.org/x/image/draw"
)

// ---------- DTO ----------

type imageRequest struct {
	Prompt         string `json:"prompt"`                    // обязательно
	Size           string `json:"size,omitempty"`            // "1:1" | "3:2" | "2:3" | "1024x1024" | ...
	ResponseFormat string `json:"response_format,omitempty"` // игнорируем, всегда приведём к URL
	AutoOpen       bool   `json:"auto_open,omitempty"`       // если true — сразу открыть в браузере
}

type imageResponse struct {
	URL  string `json:"url"`            // /static/…png
	Size string `json:"size,omitempty"` // исходный (или нормализованный) размер
	// можно добавить другие поля по желанию
}

// ---------- JSON helpers ----------

type jsonErr struct {
	Error  string `json:"error"`
	Status int    `json:"status"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, jsonErr{Error: msg, Status: status})
}

// ---------- handler ----------

func imageHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeJSONError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	if ct := r.Header.Get("Content-Type"); ct != "" && !strings.Contains(ct, "application/json") {
		writeJSONError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
		return
	}

	var in imageRequest
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		writeJSONError(w, http.StatusBadRequest, "bad JSON: "+err.Error())
		return
	}
	_ = r.Body.Close()

	in.Prompt = strings.TrimSpace(in.Prompt)
	if in.Prompt == "" {
		writeJSONError(w, http.StatusBadRequest, "prompt is required")
		return
	}
	size := strings.TrimSpace(in.Size)

	// 1) генерируем (по твоей функции)
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second) // увеличим, чтобы избежать таймаутов
	defer cancel()

	// формат ответа от провайдера нам без разницы — мы всё равно сохраним локально
	result, err := generateImage(ctx, in.Prompt, size, "url")
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "AI error: "+err.Error())
		return
	}

	// 2) сохраняем в ./static и получаем короткий URL
	// ВРЕМЕННО ОТКЛЮЧЕНО: сохранение на диск из-за проблем с правами доступа
	// publicURL, err := materializeImageToStatic(r.Context(), result)
	// if err != nil {
	// 	writeJSONError(w, http.StatusBadGateway, "save image: "+err.Error())
	// 	return
	// }
	
	// Используем оригинальный URL от OpenAI вместо локального сохранения
	publicURL := result

	// 3) опционально открываем в браузере на машине, где крутится сервер
	if in.AutoOpen {
		_ = openInBrowser("http://" + r.Host + publicURL) // best-effort; ошибки не фатальны
	}

	// 4) отдаём короткий URL
	writeJSON(w, http.StatusOK, imageResponse{
		URL:  publicURL,
		Size: size,
	})
}

// ---------- saving helpers ----------

// materializeImageToStatic принимает либо https-URL, либо data:image/png;base64,…
func materializeImageToStatic(ctx context.Context, src string) (string, error) {
	if strings.HasPrefix(src, "http://") || strings.HasPrefix(src, "https://") {
		return downloadToStatic(ctx, src)
	}
	if strings.HasPrefix(src, "data:image/png;base64,") {
		return dataURLToStatic(src)
	}
	return "", errors.New("unsupported image source format")
}

func staticDir() string {
	return "static"
}

func ensureStaticDir() error {
	return os.MkdirAll(staticDir(), 0o755)
}

func newImageName(suffix string, seed []byte) string {
	sum := sha256.Sum256(seed)
	return fmt.Sprintf("img_%d_%x%s", time.Now().Unix(), sum[:4], suffix)
}

func dataURLToStatic(dataURL string) (string, error) {
	const prefix = "data:image/png;base64,"
	b64 := strings.TrimPrefix(strings.TrimSpace(dataURL), prefix)
	return rawBase64PNGToStatic(b64)
}

func rawBase64PNGToStatic(b64 string) (string, error) {
	if err := ensureStaticDir(); err != nil {
		return "", err
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "", err
	}
	name := newImageName(".png", []byte(b64[:min(64, len(b64))]))
	path := filepath.Join(staticDir(), name)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		return "", err
	}
	return "/static/" + name, nil
}

func downloadToStatic(ctx context.Context, url string) (string, error) {
	if err := ensureStaticDir(); err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	res, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", fmt.Errorf("download http %s", res.Status)
	}
	data, err := io.ReadAll(res.Body)
	if err != nil {
		return "", err
	}

	// имя
	name := newImageName(".png", data[:min(64, len(data))])
	path := filepath.Join(staticDir(), name)

	// сохраняем «полный» PNG
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}

	// --- делаем thumbnail 512x512 ---
	img, _, err := image.Decode(bytes.NewReader(data))
	if err == nil {
		thumb := image.NewRGBA(image.Rect(0, 0, 512, 512))
		draw.CatmullRom.Scale(thumb, thumb.Bounds(), img, img.Bounds(), draw.Over, nil)
		buf := new(bytes.Buffer)
		if err := png.Encode(buf, thumb); err == nil {
			_ = os.WriteFile(filepath.Join(staticDir(), strings.TrimSuffix(name, ".png")+"_thumb.png"), buf.Bytes(), 0o644)
		}
	}

	return "/static/" + name, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ---------- open in browser (best-effort) ----------

func openInBrowser(url string) error {
	switch runtime.GOOS {
	case "darwin":
		return exec.Command("open", url).Start()
	case "linux":
		return exec.Command("xdg-open", url).Start()
	case "windows":
		// start открывается через cmd
		return exec.Command("cmd", "/c", "start", url).Start()
	default:
		return nil
	}
}
