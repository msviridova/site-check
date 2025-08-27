package main

import (
	"net/url"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

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
