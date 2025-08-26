// ai_images.go
package main

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
)

// --- helpers ---

func preview512(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 512 {
		return s[:512]
	}
	return s
}

// Конвертация строкового размера в enum SDK.
// Поддержаны стандартные квадратные форматы + безопасные маппинги
// на вертикальный/горизонтальный баннер (если ваша минорка их знает).
func toImgSize(s string) openai.ImageGenerateParamsSize {
	switch strings.TrimSpace(s) {
	case "256x256":
		return openai.ImageGenerateParamsSize1024x1024 // 256x256 больше не поддерживается
	case "512x512":
		return openai.ImageGenerateParamsSize1024x1024 // 512x512 тоже
	case "1024x1024", "":
		return openai.ImageGenerateParamsSize1024x1024
	case "1024x1536":
		return openai.ImageGenerateParamsSize1024x1536
	case "1536x1024":
		return openai.ImageGenerateParamsSize1536x1024

	// старые форматы — маппим на новые
	case "1792x1024":
		return openai.ImageGenerateParamsSize1536x1024
	case "1024x1792":
		return openai.ImageGenerateParamsSize1024x1536
	case "1792x448":
		return openai.ImageGenerateParamsSize1536x1024
	case "512x1024":
		return openai.ImageGenerateParamsSize1024x1536

	default:
		return openai.ImageGenerateParamsSize1024x1024
	}
}

// --- main ---
//
// generateImage генерирует картинку через gpt-image-1.
// size: "1024x1024" | "512x512" | "256x256" | (возможны) "1792x1024"/"1024x1792"
// responseFormat: "url" | "b64_json"
// Если сервер не вернул прямой URL, при запросе "url" вернём data:URL из base64.
func generateImage(ctx context.Context, prompt, size, responseFormat string) (string, error) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", errors.New("empty prompt")
	}
	if strings.TrimSpace(size) == "" {
		size = "1024x1024"
	}
	if strings.TrimSpace(responseFormat) == "" {
		responseFormat = "url"
	}

	// Лог вызова в ai_logs
	start := time.Now()
	aiID, _ := aiLogStart(ctx, nil, "gpt-image-1", preview512(prompt))

	// Формируем параметры генерации
	params := openai.ImageGenerateParams{
		Model:  "gpt-image-1",
		Prompt: prompt,
		Size:   toImgSize(size),
		// В этой версии SDK параметр response_format в params отсутствует.
		// API часто возвращает только b64_json — обработаем это ниже.
	}

	// Вызов API
	resp, err := aiClient.Images.Generate(ctx, params)
	if err != nil {
		aiLogFinish(ctx, aiID, "", err.Error(), nil, time.Since(start))
		return "", err
	}
	aiLogFinish(ctx, aiID, "", "", nil, time.Since(start))

	// Разбор ответа
	if len(resp.Data) == 0 {
		return "", errors.New("empty image response: no data")
	}
	b64 := strings.TrimSpace(resp.Data[0].B64JSON)
	url := strings.TrimSpace(resp.Data[0].URL)

	// Если просят base64 — отдаём строго b64_json
	if strings.ToLower(responseFormat) == "b64_json" {
		if b64 == "" {
			return "", errors.New("image API returned empty b64_json")
		}
		return b64, nil
	}

	// Иначе хотели URL. Если его нет — вернём data-URL, пригодный для <img src="...">
	if url != "" {
		return url, nil
	}
	if b64 != "" {
		return "data:image/png;base64," + b64, nil
	}

	return "", errors.New("image API returned neither url nor b64_json")
}
