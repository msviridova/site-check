package main

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

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
