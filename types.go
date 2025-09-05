package main

import "time"

// ====== запрос для /creative ======
type CreativeRequest struct {
	Kind             string `json:"kind"`               // "text" | "graphic"
	SiteText         string `json:"site_text"`          // ОБЯЗАТЕЛЕН
	SiteURL          string `json:"site_url,omitempty"` // опционально, только как контекст
	Goal             string `json:"goal,omitempty"`
	Audience         string `json:"audience,omitempty"`
	Geo              string `json:"geo,omitempty"`
	OfferConstraints string `json:"offer_constraints,omitempty"`
	BrandOverrides   string `json:"brand_overrides,omitempty"`
	PreferredAspect  string `json:"preferred_aspect,omitempty"` // "1x1" | "4x1" | "1x2"
}

// ====== ответ для /creative ======
type CreativeResponse struct {
	Kind   string `json:"kind"`   // "text" | "graphic"
	Lang   string `json:"lang"`   // "ru"
	Source string `json:"source"` // "ai" | "ai_error"

	// Для текстовых креативов (все типы сразу)
	Keywords  []string  `json:"keywords,omitempty"`
	Negatives []string  `json:"negatives,omitempty"`
	Ads       []AdBlock `json:"ads,omitempty"`

	// Для графических креативов
	Graphic *GraphicPlan `json:"graphic,omitempty"`
}

// ====== классификация сайта (/classify) ======

type classifyRequest struct {
	URL string `json:"url"`
}

type classifyResponse struct {
	Summary string `json:"summary"`
	Lang    string `json:"lang"`
	Source  string `json:"source"` // "ai" | "heuristic" | "ai_quota" | "ai_error"

	// Брендинг / стиль
	Brand      string `json:"brand,omitempty"`
	StyleNotes string `json:"style_notes,omitempty"`

	// Палитра (расширенная)
	MainColorsHex       []string `json:"main_colors_hex,omitempty"`
	AdditionalColorsHex []string `json:"additional_colors_hex,omitempty"`
	BackgroundColorHex  string   `json:"background_color_hex,omitempty"`
	AccentPrimaryHex    string   `json:"accent_primary_hex,omitempty"`
	AccentSecondaryHex  string   `json:"accent_secondary_hex,omitempty"`
}

// ====== объявления (для text_type = "ads") ======
type AdLink struct {
	URL   string `json:"url"`
	Title string `json:"title"`
	Desc  string `json:"desc"`
}

type AdBlock struct {
	ID      string   `json:"id"`
	Header  string   `json:"header"`
	Text    string   `json:"text"`
	Links   []AdLink `json:"links"`
	Details []string `json:"details"`
}

// ====== входные опции для графики ======
type GraphicInputOpts struct {
	Goal             string
	Audience         string
	Geo              string
	OfferConstraints string
	BrandOverrides   string
}

// ====== графический план (ответ для Kind="graphic") ======
type GraphicPlan struct {
	Brand struct {
		Name               string   `json:"name"`
		ExtractedColorsHex []string `json:"extracted_colors_hex"`
		StyleNotes         string   `json:"style_notes"`
		Assumptions        string   `json:"assumptions"`
	} `json:"brand"`

	Concepts []struct {
		Name       string `json:"name"`
		Rationale  string `json:"rationale"`
		VisualPlan string `json:"visual_plan"`

		ImagePrompts struct {
			Sq1x1 string `json:"1x1"`
			Ar4x1 string `json:"4x1"`
			Ar1x2 string `json:"1x2"`
		} `json:"image_prompts"`

		// ⚡️ сюда будем писать URL уже сгенерированных картинок
		ImageURLs struct {
			Sq1x1 string `json:"url_1x1,omitempty"`
			Ar4x1 string `json:"url_4x1,omitempty"`
			Ar1x2 string `json:"url_1x2,omitempty"`
		} `json:"image_urls,omitempty"`

		Negatives       string `json:"negatives"`
		GeneratorParams struct {
			Stylize int `json:"stylize"`
			Chaos   int `json:"chaos"`
			Seed    int `json:"seed"`
		} `json:"generator_params"`

		AdCopyRu struct {
			Headline string `json:"headline"`
			Body     string `json:"body"`
			CTA      string `json:"cta"`
		} `json:"ad_copy_ru"`
	} `json:"concepts"`

	SafetyChecklist []string `json:"safety_checklist"`
}

// === image generation ===

type ImageRequest struct {
	Prompt         string `json:"prompt"`
	Size           string `json:"size,omitempty"`            // "1:1", "3:2", "2:3"
	ResponseFormat string `json:"response_format,omitempty"` // "url" или "b64_json"
}

type ImageResponse struct {
	Prompt         string `json:"prompt"`
	Size           string `json:"size"`
	ResponseFormat string `json:"response_format"`
	URL            string `json:"url,omitempty"`
	B64JSON        string `json:"b64_json,omitempty"`
	Lang           string `json:"lang"`
	Source         string `json:"source"`
	Error          string `json:"error,omitempty"`
}

// ==== PROMPTS (для хранения в БД и отдачи наружу) ====

type Prompt struct {
	ID          int       `json:"id" db:"id"`
	KeyName     string    `json:"key_name" db:"key_name"`
	Locale      string    `json:"locale" db:"locale"`
	Version     int       `json:"version" db:"version"`
	Description string    `json:"description" db:"description"`
	Text        string    `json:"text" db:"text"`
	IsActive    bool      `json:"is_active" db:"is_active"`
	UpdatedBy   string    `json:"updated_by" db:"updated_by"`
	UpdatedAt   time.Time `json:"updated_at" db:"updated_at"`
}

// ==== запрос/ответ для API ====

type PromptRequest struct {
	KeyName string `json:"key_name"` // какой промпт запрашиваем
	Locale  string `json:"locale"`   // язык (например "ru" или "en")
	Version int    `json:"version"`  // версия (можно оставить 0 = последняя)
}

type PromptResponse struct {
	KeyName string `json:"key_name"`
	Locale  string `json:"locale"`
	Version int    `json:"version"`
	Text    string `json:"text"`
}
