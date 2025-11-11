package main

import (
	"database/sql"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

// App представляет основное приложение со всеми зависимостями
type App struct {
	DB          *sql.DB
	Store       Store
	Config      *Config
	promptCache struct {
		mu  sync.RWMutex
		ttl time.Duration
		m   map[string]cachedPrompt
	}
}

type cachedPrompt struct {
	prompt  *Prompt
	expires time.Time
}

// NewApp создает новое приложение с инициализированными зависимостями
func NewApp() (*App, error) {
	// Загружаем конфигурацию
	config, err := NewConfig()
	if err != nil {
		return nil, err
	}

	// Инициализируем БД
	db, err := openDB()
	if err != nil {
		return nil, err
	}

	app := &App{
		DB:     db,
		Config: config,
	}
	app.Store = NewSQLStore(db)
	app.promptCache.ttl = config.PromptCacheTTL
	app.promptCache.m = make(map[string]cachedPrompt)

	// Логируем информацию о конфигурации
	app.logConfig()

	return app, nil
}

// Close закрывает все ресурсы приложения
func (app *App) Close() error {
	if app.DB != nil {
		return app.DB.Close()
	}
	return nil
}

// logConfig логирует информацию о конфигурации
func (app *App) logConfig() {
	apiKey := getEnv("OPENAI_API_KEY")
	if apiKey == "" {
		logWarn("OPENAI_API_KEY is empty — AI will fallback to heuristic", nil)
	} else {
		logInfo("OPENAI_API_KEY detected", map[string]interface{}{"length": len(apiKey)})
	}
	logInfo("boot configuration", map[string]interface{}{
		"use_ai":  app.Config.UseAI,
		"model":   app.Config.ModelName,
		"key_set": apiKey != "",
		"key":     maskKey(apiKey),
	})
}

// openDB открывает соединение с БД
func openDB() (*sql.DB, error) {
	dsn := os.Getenv("DATABASE_URL")
	if strings.TrimSpace(dsn) == "" {
		log.Fatal("DATABASE_URL is empty")
	}

	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, err
	}

	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(10)
	db.SetConnMaxLifetime(30 * time.Minute)

	if err := db.Ping(); err != nil {
		return nil, err
	}

	return db, nil
}
