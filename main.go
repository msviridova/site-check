package main

import (
	"errors"
	"log"
	"net/http"
	"strings"
)

func main() {
	// Инициализация приложения
	app, err := NewApp()
	if err != nil {
		log.Fatal("Failed to initialize app:", err)
	}
	defer app.Close()

	// Маршруты
	mux := http.NewServeMux()
	mux.HandleFunc("/classify", app.classifyHandler)
	mux.HandleFunc("/creative", app.creativeHandler)
	mux.HandleFunc("/image", app.imageHandler)
	mux.HandleFunc("/prompts", app.promptHandler)
	mux.HandleFunc("/prompts/update", app.promptUpdateHandler)
	mux.HandleFunc("/prompts/list", app.promptListHandler)
	mux.HandleFunc("/logs/recent", app.logsHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	if dir := strings.TrimSpace(app.Config.StaticDir); dir != "" {
		fs := http.FileServer(http.Dir(dir))
		mux.Handle("/static/", http.StripPrefix("/static/", fs))
	}

	srv := &http.Server{
		Addr:              app.Config.ListenAddr,
		Handler:           mux,
		ReadHeaderTimeout: app.Config.ServerReadHeaderTimeout,
	}

	logInfo("server starting", map[string]interface{}{"addr": app.Config.ListenAddr})
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logError("server shutdown", map[string]interface{}{"error": err.Error()})
		log.Fatal(err)
	}
}
