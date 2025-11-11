package main

import (
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

// Config содержит конфигурацию приложения
type Config struct {
	// AI настройки
	UseAI     bool
	ModelName string
	AIClient  openai.Client

	// HTTP сервер/клиент
	HTTPClient              *http.Client
	ListenAddr              string
	StaticDir               string
	ServerReadHeaderTimeout time.Duration

	// Таймауты для запросов к AI
	ClassifyTimeout time.Duration
	CreativeTimeout time.Duration
	AITimeout       time.Duration
	HTTPTimeout     time.Duration

	// Лимиты
	ClassifySiteTextMax     int
	CreativeSiteTextMax     int
	GraphicSiteTextMax      int
	ImageConceptMaxBytes    int
	ImageAdditionalMaxBytes int

	// Кэш
	PromptCacheTTL time.Duration
}

// NewConfig создает конфигурацию из environment variables
func NewConfig() (*Config, error) {
	apiKey := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	useAI := strings.ToLower(os.Getenv("USE_AI")) == "true"
	modelName := os.Getenv("MODEL_NAME")
	if modelName == "" {
		modelName = "gpt-4.1"
	}

	listenAddr := os.Getenv("HTTP_ADDR")
	if listenAddr == "" {
		listenAddr = ":8080"
	}

	staticDir := os.Getenv("STATIC_DIR")
	if staticDir == "" {
		staticDir = "./static"
	}

	// AI client
	aiClient := openai.NewClient(option.WithAPIKey(apiKey))

	httpTimeout := 10 * time.Second
	if timeoutStr := os.Getenv("HTTP_TIMEOUT"); timeoutStr != "" {
		if d, err := time.ParseDuration(timeoutStr); err == nil {
			httpTimeout = d
		}
	}

	promptCacheTTL := 5 * time.Minute
	if ttl := os.Getenv("PROMPT_CACHE_TTL"); ttl != "" {
		if d, err := time.ParseDuration(ttl); err == nil {
			promptCacheTTL = d
		}
	}

	readHeaderTimeout := 5 * time.Second
	if rht := os.Getenv("SERVER_READ_HEADER_TIMEOUT"); rht != "" {
		if d, err := time.ParseDuration(rht); err == nil {
			readHeaderTimeout = d
		}
	}

	return &Config{
		UseAI:                   useAI,
		ModelName:               modelName,
		AIClient:                aiClient,
		HTTPClient:              &http.Client{Timeout: httpTimeout},
		ListenAddr:              listenAddr,
		StaticDir:               staticDir,
		ServerReadHeaderTimeout: readHeaderTimeout,

		ClassifyTimeout: 45 * time.Second,
		CreativeTimeout: 120 * time.Second,
		AITimeout:       120 * time.Second,
		HTTPTimeout:     httpTimeout,

		ClassifySiteTextMax:     12000,
		CreativeSiteTextMax:     10000,
		GraphicSiteTextMax:      8000,
		ImageConceptMaxBytes:    4000,
		ImageAdditionalMaxBytes: 512,
		PromptCacheTTL:          promptCacheTTL,
	}, nil
}

// maskKey маскирует API-ключ для логов
func maskKey(s string) string {
	if len(s) <= 8 {
		return s
	}
	return s[:4] + "…" + s[len(s)-4:]
}

// getEnv получает значение environment variable
func getEnv(k string) string {
	return strings.TrimSpace(os.Getenv(k))
}
