// ai.go
package main

import "github.com/openai/openai-go/v2"

// Глобальный клиент OpenAI, инициализируется в main.go
var aiClient openai.Client

// CompletionUsageLike — компактный аналог usage из OpenAI для логирования в БД.
type CompletionUsageLike struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}
