package main

import (
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

// --- brand / colors / style ---

func extractBrand(u *url.URL, html string) string {
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(html))
	if err == nil {
		// 1) og:site_name
		if v, ok := doc.Find(`meta[property="og:site_name"]`).Attr("content"); ok {
			v = strings.TrimSpace(v)
			if v != "" {
				return v
			}
		}
		// 2) application-name
		if v, ok := doc.Find(`meta[name="application-name"]`).Attr("content"); ok {
			v = strings.TrimSpace(v)
			if v != "" {
				return v
			}
		}
		// 3) title
		if t := strings.TrimSpace(doc.Find("title").First().Text()); t != "" {
			// часто в title есть «Бренд — описание»
			parts := strings.Split(t, "—")
			if len(parts) == 1 {
				parts = strings.Split(t, "-")
			}
			lead := strings.TrimSpace(parts[0])
			if lead != "" && len(lead) <= 40 {
				return lead
			}
			return t
		}
	}
	// 4) из домена
	host := u.Hostname()
	if host == "" {
		host = u.Host
	}
	seg := strings.Split(host, ".")
	if len(seg) >= 2 {
		return strings.Title(seg[len(seg)-2]) // второй уровень домена
	}
	return host
}

var hexRe = regexp.MustCompile(`(?i)#([0-9a-f]{6}|[0-9a-f]{3})\b`)

func expand3to6(h string) string {
	// "#abc" -> "#aabbcc"
	if len(h) == 4 {
		return "#" + strings.Repeat(string(h[1]), 2) +
			strings.Repeat(string(h[2]), 2) +
			strings.Repeat(string(h[3]), 2)
	}
	return strings.ToLower(h)
}

func extractColorsHex(html string) []string {
	// собираем все hex-цвета из HTML + meta theme-color
	var all []string
	all = append(all, hexRe.FindAllString(html, -1)...)

	// meta theme-color
	if idx := strings.Index(strings.ToLower(html), `name="theme-color"`); idx >= 0 {
		// грубый поиск следующего hex рядом
		near := html[idx:]
		if m := hexRe.FindString(near); m != "" {
			all = append(all, m)
		}
	}

	if len(all) == 0 {
		return nil
	}

	// нормализуем и считаем частоты
	cnt := map[string]int{}
	for _, c := range all {
		c = expand3to6(strings.ToLower(c))
		if len(c) == 7 {
			cnt[c]++
		}
	}

	// отобрать топ-3
	type kv struct {
		k string
		v int
	}
	var arr []kv
	for k, v := range cnt {
		arr = append(arr, kv{k, v})
	}
	// простая сортировка по частоте (пузырьком не надо — хватит selection)
	for i := 0; i < len(arr); i++ {
		for j := i + 1; j < len(arr); j++ {
			if arr[j].v > arr[i].v {
				arr[i], arr[j] = arr[j], arr[i]
			}
		}
	}

	limit := 3
	if len(arr) < limit {
		limit = len(arr)
	}
	out := make([]string, 0, limit)
	seen := map[string]bool{}
	for _, it := range arr[:limit] {
		if !seen[it.k] {
			out = append(out, it.k)
			seen[it.k] = true
		}
	}
	return out
}

func hexToRGB(h string) (float64, float64, float64, bool) {
	h = strings.TrimPrefix(h, "#")
	if len(h) != 6 {
		return 0, 0, 0, false
	}
	rv, err1 := strconv.ParseUint(h[0:2], 16, 8)
	gv, err2 := strconv.ParseUint(h[2:4], 16, 8)
	bv, err3 := strconv.ParseUint(h[4:6], 16, 8)
	if err1 != nil || err2 != nil || err3 != nil {
		return 0, 0, 0, false
	}
	// нормализуем 0..1
	return float64(rv) / 255.0, float64(gv) / 255.0, float64(bv) / 255.0, true
}

func luminance(r, g, b float64) float64 {
	// относительная яркость (sRGB)
	return 0.2126*r + 0.7152*g + 0.0722*b
}

func saturation(r, g, b float64) float64 {
	max := r
	if g > max {
		max = g
	}
	if b > max {
		max = b
	}
	min := r
	if g < min {
		min = g
	}
	if b < min {
		min = b
	}
	if max == 0 {
		return 0
	}
	return (max - min) / max
}

func deriveStyleNotes(colors []string, html string) string {
	if len(colors) == 0 {
		// попробуем meta color-scheme
		ls := strings.ToLower(html)
		if strings.Contains(ls, "color-scheme") && strings.Contains(ls, "dark") {
			return "тёмная палитра"
		}
		return ""
	}

	// берём самый частый как доминирующий
	r, g, b, ok := hexToRGB(colors[0])
	if !ok {
		return ""
	}
	l := luminance(r, g, b)
	s := saturation(r, g, b)

	var notes []string
	if l < 0.35 {
		notes = append(notes, "тёмная палитра")
	} else if l > 0.75 {
		notes = append(notes, "очень светлая палитра")
	} else {
		notes = append(notes, "светлая палитра")
	}
	if s > 0.6 {
		notes = append(notes, "яркие акценты")
	} else if s < 0.2 {
		notes = append(notes, "приглушённые тона")
	}
	return strings.Join(notes, ", ")
}
