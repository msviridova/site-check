package main

// ==== структуры для входа/выхода ====

type classifyRequest struct {
	URL string `json:"url"`
}

type classifyResponse struct {
	Summary            string   `json:"summary"`
	Lang               string   `json:"lang"`
	Source             string   `json:"source"` // "ai" / "heuristic" / "ai_quota" / "ai_error"
	Brand              string   `json:"brand,omitempty"`
	ExtractedColorsHex []string `json:"extracted_colors_hex,omitempty"`
	StyleNotes         string   `json:"style_notes,omitempty"`
	Keywords           []string `json:"keywords,omitempty"`
	NegativeKeywords   []string `json:"negative_keywords,omitempty"`
}
