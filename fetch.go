// fetch.go
package main

import (
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

func fetchHTML(ctx context.Context, url string, client *http.Client) (string, error) {
	if client == nil {
		// fallback: создаем клиент по умолчанию
		jar, _ := cookiejar.New(nil)
		client = &http.Client{
			Jar:     jar,
			Timeout: 15 * time.Second,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				if len(via) >= 10 {
					return errors.New("too many redirects")
				}
				req.Header.Set("User-Agent", via[0].Header.Get("User-Agent"))
				return nil
			},
		}
	}
	// 1) пробуем «как браузер»
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7")
	req.Header.Set("Accept-Encoding", "gzip")
	req.Header.Set("Cache-Control", "no-cache")

	res, err := client.Do(req)
	if err == nil && res != nil {
		defer res.Body.Close()
		if res.StatusCode >= 200 && res.StatusCode < 300 {
			var reader io.Reader = res.Body
			if strings.Contains(strings.ToLower(res.Header.Get("Content-Encoding")), "gzip") {
				gz, gzErr := gzip.NewReader(res.Body)
				if gzErr == nil {
					defer gz.Close()
					reader = gz
				}
			}
			b, _ := io.ReadAll(reader)
			return string(b), nil
		}
		// если явный запрет — пойдём на фолбэк
		if res.StatusCode != 401 && res.StatusCode != 403 && res.StatusCode != 503 {
			return "", fmt.Errorf("non-2xx status: %s", res.Status)
		}
	}

	// 2) фолбэк через text-render proxy (тянет только видимый текст)
	// https://r.jina.ai/http://<host> — удобный публичный рендер для чтения страниц
	fb := "https://r.jina.ai/http://" + strings.TrimPrefix(strings.TrimPrefix(url, "https://"), "http://")
	req2, _ := http.NewRequestWithContext(ctx, http.MethodGet, fb, nil)
	req2.Header.Set("User-Agent", "curl/8.0")
	req2.Header.Set("Accept", "text/plain")
	res2, err2 := client.Do(req2)
	if err2 != nil {
		return "", fmt.Errorf("fetch failed: %v", err2)
	}
	defer res2.Body.Close()
	if res2.StatusCode < 200 || res2.StatusCode >= 300 {
		return "", fmt.Errorf("fallback non-2xx status: %s", res2.Status)
	}
	b, _ := io.ReadAll(res2.Body)
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
