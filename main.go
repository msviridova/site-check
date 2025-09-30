package main

import (
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

// ==== глобальные настройки ====

var (
	// переключатель использования AI
	useAI = strings.ToLower(os.Getenv("USE_AI")) == "true"

	// модель — можно вынести в ENV при желании
	modelName = "gpt-4.1"

	// общий http‑клиент
	httpClient = &http.Client{Timeout: 10 * time.Second}
)

// maskKey маскирует API‑ключ для логов
func maskKey(s string) string {
	if len(s) <= 8 {
		return s
	}
	return s[:4] + "…" + s[len(s)-4:]
}

func getEnv(k string) string { return strings.TrimSpace(os.Getenv(k)) }

func main() {
	// ИНИЦИАЛИЗАЦИЯ глобального aiClient (ОБЯЗАТЕЛЬНО: в ai.go должно быть `var aiClient *openai.Client`)
	aiClient = openai.NewClient(option.WithAPIKey(strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))))

	// Логи окружения
	if getEnv("OPENAI_API_KEY") == "" {
		log.Println("WARN: OPENAI_API_KEY is empty — AI will fallback to heuristic")
	} else {
		log.Printf("INFO: OPENAI_API_KEY detected (len=%d)\n", len(getEnv("OPENAI_API_KEY")))
	}
	log.Printf("BOOT: USE_AI=%v MODEL=%s KEY_SET=%t KEY=%s",
		useAI, modelName, getEnv("OPENAI_API_KEY") != "", maskKey(getEnv("OPENAI_API_KEY")))

	// БД
	db = mustOpenDB()
	defer db.Close()

	// Маршруты
	mux := http.NewServeMux()
	mux.HandleFunc("/classify", classifyHandler)
	mux.HandleFunc("/creative", creativeHandler)
	mux.HandleFunc("/image", imageHandler) // <— вот это обязательно
	mux.HandleFunc("/prompts", promptHandler)
	mux.HandleFunc("/prompts/list", promptListHandler)
	mux.HandleFunc("/logs/recent", logsHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	fs := http.FileServer(http.Dir("./static"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))

	srv := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Println("listening on :8080")
	log.Fatal(srv.ListenAndServe())
}
