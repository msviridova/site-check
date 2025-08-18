package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/openai/openai-go/v2"
)

// ==== структуры для входа/выхода ====

type classifyRequest struct {
	URL string `json:"url"`
}

type classifyResponse struct {
	Summary          string   `json:"summary"`
	Lang             string   `json:"lang"`
	Source           string   `json:"source"` // "ai" / "heuristic" / "ai_quota" / "ai_error"
	Keywords         []string `json:"keywords,omitempty"`
	NegativeKeywords []string `json:"negative_keywords,omitempty"`
}

// ==== HTTP-обработчик ====

func classifyHandler(w http.ResponseWriter, r *http.Request) {
	// 1) принимаем только POST
	if r.Method != http.MethodPost {
		http.Error(w, "use POST", http.StatusMethodNotAllowed)
		return
	}

	// 2) читаем JSON
	var req classifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad JSON", http.StatusBadRequest)
		return
	}

	// 3) валидируем URL
	raw := strings.TrimSpace(req.URL)
	if raw == "" {
		http.Error(w, "url is required", http.StatusBadRequest)
		return
	}
	u, err := url.ParseRequestURI(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		http.Error(w, "invalid url", http.StatusBadRequest)
		return
	}

	log.Printf("useAI=%v url=%s", useAI, u.String())

	// 4) общий таймаут на работу хэндлера
	ctx, cancel := context.WithTimeout(r.Context(), 12*time.Second)
	defer cancel()

	// 5) скачиваем HTML
	html, err := fetchHTML(ctx, u.String())
	if err != nil {
		http.Error(w, "fetch failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	// 6) извлекаем видимый текст
	text := extractVisibleText(html)
	log.Printf("extracted text length: %d", len(text))

	// === НОВОЕ: если текста мало — пробуем ИИ по домену (и title/meta), иначе фолбэк ===
	if len(strings.TrimSpace(text)) < 40 {
		brief := fallbackSummary(u, html) // title/meta/host
		if useAI {
			// соберём небольшой вход для модели
			shortInput := "Домен: " + u.Hostname()
			if b := strings.TrimSpace(brief); b != "" {
				shortInput += "\nTitle/Meta: " + b
			}

			sum, kws, negs, aiErr := summarizeWithAI(ctx, shortInput)
			log.Printf("AI (short-text) finished, err=%v", aiErr)
			if aiErr == nil && strings.TrimSpace(sum) != "" {
				resp := classifyResponse{
					Summary:          sum,
					Lang:             "ru",
					Source:           "ai",
					Keywords:         kws,
					NegativeKeywords: negs,
				}
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				_ = json.NewEncoder(w).Encode(resp)
				return
			}
			log.Println("AI short-text failed → fallback to heuristic")
		}

		// эвристический фолбэк
		summary := brief
		if strings.TrimSpace(summary) == "" {
			summary = "Веб-сайт компании/сервиса " + u.Hostname()
		}
		resp := classifyResponse{
			Summary: summary,
			Lang:    "ru",
			Source:  "heuristic",
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(resp)
		return
	}

	// === Текста достаточно: обычная логика ===
	var (
		summary string
		source  string
		kws     []string
		negs    []string
	)

	if useAI {
		source = "ai"
		sum, kk, nn, aiErr := summarizeWithAI(ctx, text)
		log.Printf("AI call finished, err=%v", aiErr)
		if aiErr != nil || strings.TrimSpace(sum) == "" {
			log.Println("AI failed or empty → fallback to heuristic")
			summary = heuristicSummarize(text)
			source = "heuristic"
		} else {
			summary, kws, negs = sum, kk, nn
		}
	} else {
		summary = heuristicSummarize(text)
		source = "heuristic"
	}

	// стоп-фолбэк: не отдаём пустую строку
	if strings.TrimSpace(summary) == "" {
		log.Println("summary is empty → using title/meta/host fallback")
		summary = fallbackSummary(u, html)
		if strings.TrimSpace(summary) == "" {
			summary = "Не удалось определить тематику сайта"
		}
	}

	// 8) отвечаем JSON
	resp := classifyResponse{
		Summary:          summary,
		Lang:             "ru",
		Source:           source,
		Keywords:         kws,
		NegativeKeywords: negs,
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}

// ==== HTTP-клиент ====

var httpClient = &http.Client{
	Timeout: 10 * time.Second,
}

var aiClient = openai.NewClient()

var useAI = strings.ToLower(os.Getenv("USE_AI")) == "true"

var modelName = "gpt-3.5-turbo" // как и было

func maskKey(s string) string {
	if len(s) <= 8 {
		return s
	}
	return s[:4] + "…" + s[len(s)-4:]
}

// ==== загрузка HTML ====

func fetchHTML(ctx context.Context, target string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "site-check/1.0 (+learning-go)")

	res, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return "", errors.New("non-2xx status: " + res.Status)
	}

	b, err := io.ReadAll(io.LimitReader(res.Body, 2<<20)) // 2 MiB лимит
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ==== извлечение видимого текста ====

func extractVisibleText(html string) string {
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(html))
	if err != nil {
		return ""
	}

	// убрать шумные блоки
	doc.Find("script, style, noscript, nav, header, footer, template, svg, iframe, aside").Remove()

	// берём title и meta description в приоритет
	title := strings.TrimSpace(doc.Find("title").First().Text())
	metaDesc := ""
	doc.Find(`meta[name="description"]`).Each(func(_ int, s *goquery.Selection) {
		if v, ok := s.Attr("content"); ok {
			metaDesc = strings.TrimSpace(v)
		}
	})

	clean := func(s string) string {
		// нормализуем пробелы
		s = strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
		return s
	}

	isNoisy := func(s string) bool {
		ls := strings.ToLower(s)
		// явные признаки JSON/шаблонов/тех. мусора
		if strings.Contains(ls, "{") && strings.Contains(ls, "}") {
			return true
		}
		if strings.Contains(ls, "[") && strings.Contains(ls, "]") {
			return true
		}
		if strings.Contains(ls, "widgets") || strings.Contains(ls, "cookie") || strings.Contains(ls, "tracking") {
			return true
		}
		// слишком много «небуквенных» символов → похоже на код
		var non, letters int
		for _, r := range ls {
			if (r >= 'a' && r <= 'z') || (r >= 'а' && r <= 'я') || r == 'ё' {
				letters++
			} else if r != ' ' {
				non++
			}
		}
		return letters > 0 && float64(non)/float64(letters+1) > 0.7
	}

	var parts []string
	if title != "" && !isNoisy(title) {
		parts = append(parts, clean(title))
	}
	if metaDesc != "" && !isNoisy(metaDesc) {
		parts = append(parts, clean(metaDesc))
	}

	// собираем важный текст страницы
	doc.Find("h1, h2, h3, p, li").Each(func(_ int, s *goquery.Selection) {
		t := clean(s.Text())
		if t != "" && !isNoisy(t) && len(t) >= 10 {
			parts = append(parts, t)
		}
	})

	text := strings.Join(parts, " ")
	if len(text) > 20000 {
		text = text[:20000]
	}
	return text
}

// ==== простая эвристика ====

func heuristicSummarize(text string) string {
	if text == "" {
		return "Информация о сайте не определена"
	}
	l := strings.ToLower(text)

	// быстрые правила для e‑commerce/marketplace
	hasAny := func(s string, keys ...string) bool {
		for _, k := range keys {
			if strings.Contains(s, k) {
				return true
			}
		}
		return false
	}

	switch {
	case hasAny(l, "маркетплейс", "продавцы", "продавцов", "отзывы", "рейтинг") && hasAny(l, "товар", "каталог", "купить", "цены", "доставка"):
		return "Маркетплейс: товары от разных продавцов."
	case hasAny(l, "яндекс маркет", "market.yandex", "яндекс‑маркет", "yandex market"):
		return "Маркетплейс: Яндекс Маркет (онлайн‑покупки)."
	case hasAny(l, "каталог", "товар", "купить", "заказать", "цены", "доставка", "корзина"):
		return "Интернет‑магазин (каталог товаров, покупки онлайн)."
	case hasAny(l, "доставка еды", "пицца", "суши", "роллы", "бургер", "заказ еды"):
		return "Доставка готовой еды."
	case hasAny(l, "услуги", "заказать услугу", "портфолио", "наши услуги"):
		return "Сайт компании‑услугодателя."
	}

	// иначе — берём первое осмысленное предложение
	isNoisySent := func(s string) bool {
		ss := strings.ToLower(strings.TrimSpace(s))
		if ss == "" {
			return true
		}
		if strings.Contains(ss, "{") || strings.Contains(ss, "}") || strings.Contains(ss, "widgets") {
			return true
		}
		// длина «в окно» и не сплошной мусор
		if len(ss) < 30 || len(ss) > 220 {
			return true
		}
		return false
	}

	// пробуем найти конец первого предложения
	end := strings.IndexAny(text, ".!?…")
	cand := text
	if end > 0 {
		cand = strings.TrimSpace(text[:end+1])
	}
	if !isNoisySent(cand) {
		return "Краткое описание по тексту сайта: " + cand
	}

	// если первое предложение шумное — ищем дальше
	sentences := splitSentences(text)
	for _, s := range sentences {
		if !isNoisySent(s) {
			return "Краткое описание по тексту сайта: " + s
		}
	}

	// последний фолбэк
	s := strings.Join(strings.Fields(text), " ")
	if len(s) > 180 {
		s = s[:180] + "…"
	}
	return "Краткое описание по тексту сайта: " + s
}

// простенький сплиттер предложений
func splitSentences(s string) []string {
	var out []string
	start := 0
	for i, r := range s {
		if r == '.' || r == '!' || r == '?' || r == '…' {
			part := strings.TrimSpace(s[start : i+1])
			if part != "" {
				out = append(out, part)
			}
			start = i + 1
		}
	}
	// хвост без финальной точки
	if start < len(s) {
		tail := strings.TrimSpace(s[start:])
		if tail != "" {
			out = append(out, tail)
		}
	}
	return out
}

func summarizeWithAI(ctx context.Context, text string) (string, []string, []string, error) {
	// поджимаем вход: модели не нужен весь роман
	if len(text) > 4000 {
		text = text[:4000]
	}

	// создаём «дочерний» контекст с небольшим таймаутом,
	// чтобы ИИ не подвесил наш хэндлер
	cctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	// Просим СТРОГО JSON (чтобы удобно парсить в поля ответа)
	prompt := `Ты — сервис классификации сайтов.

1) Кратко, одной деловой фразой по-русски опиши тематику сайта (сфера/услуга/товар и, если явно есть, город/бренд).
   Не добавляй лишних слов, без пояснений, без ссылок.

2) Сгенерируй список ключевых слов и фраз для запуска рекламы в Яндекс.Директ (30–40 штук, только по этому контенту).

3) Сформируй список минус-слов (30–50), чтобы отсеять нерелевантные запросы.

Верни СТРОГО валидный JSON ровно такой структуры (без пояснений снаружи):
{
  "summary": "краткое описание одной фразой",
  "keywords": ["...", "..."],
  "negative_keywords": ["...", "..."]
}

Контент сайта:
` + text

	resp, err := aiClient.Chat.Completions.New(cctx, openai.ChatCompletionNewParams{
		Model: "gpt-3.5-turbo", // как было раньше
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		// ответ длиннее — увеличим лимит
		MaxTokens:   openai.Int(800),
		Temperature: openai.Float(0.2),
		Seed:        openai.Int(42),
	})
	if err != nil {
		return "", nil, nil, err
	}
	if len(resp.Choices) == 0 {
		return "", nil, nil, errors.New("no choices from AI")
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	if raw == "" {
		return "", nil, nil, errors.New("empty AI response")
	}

	// временная структура для парсинга JSON
	var tmp struct {
		Summary          string   `json:"summary"`
		Keywords         []string `json:"keywords"`
		NegativeKeywords []string `json:"negative_keywords"`
	}
	if jerr := json.Unmarshal([]byte(raw), &tmp); jerr != nil {
		// если пришёл невалидный JSON — вернём хотя бы summary как текст,
		// списки оставим пустыми (эвристика всё равно подстрахует)
		return raw, nil, nil, nil
	}

	return strings.TrimSpace(tmp.Summary), tmp.Keywords, tmp.NegativeKeywords, nil
}

func fallbackSummary(u *url.URL, html string) string {
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(html))
	if err == nil {
		if t := strings.TrimSpace(doc.Find("title").First().Text()); t != "" {
			return "Краткое описание по тексту сайта: " + t
		}
		if md, ok := doc.Find(`meta[name="description"]`).Attr("content"); ok {
			md = strings.TrimSpace(md)
			if md != "" {
				return "Краткое описание по тексту сайта: " + md
			}
		}
	}
	host := u.Hostname()
	if host == "" {
		host = u.Host
	}
	if host != "" {
		return "Сайт: " + host
	}
	return "Не удалось определить тематику сайта"
}

// ==== точка входа ====

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Println("WARN: OPENAI_API_KEY is empty — AI will fallback to heuristic")
	} else {
		log.Printf("INFO: OPENAI_API_KEY detected (len=%d)\n", len(os.Getenv("OPENAI_API_KEY")))
	}
	log.Printf("BOOT: USE_AI=%v MODEL=%s KEY_SET=%t KEY=%s",
		useAI, modelName, os.Getenv("OPENAI_API_KEY") != "", maskKey(os.Getenv("OPENAI_API_KEY")))
	mux := http.NewServeMux()
	mux.HandleFunc("/classify", classifyHandler)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Println("listening on :8080")
	log.Fatal(srv.ListenAndServe())
}
