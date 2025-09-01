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

// ratioToSize переводит "1:1" / "3:2" / "2:3" в валидные размеры API.
// Если пришло уже пиксельное значение — возвращаем как есть.
// Неподдержанное -> "1024x1024".
func ratioToSize(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "1:1", "1x1":
		return "1024x1024"
	case "3:2", "4:3", "landscape":
		return "1536x1024" // горизонталь
	case "2:3", "3:4", "portrait":
		return "1024x1536" // вертикаль
	case "auto":
		return "1024x1024"
	default:
		if strings.Contains(s, "x") {
			return s
		}
		return "1024x1024"
	}
}

// Конвертация строкового размера в enum SDK.
func toImgSize(s string) openai.ImageGenerateParamsSize {
	switch strings.TrimSpace(s) {
	case "1024x1024":
		return openai.ImageGenerateParamsSize1024x1024
	case "1024x1536":
		return openai.ImageGenerateParamsSize1024x1536
	case "1536x1024":
		return openai.ImageGenerateParamsSize1536x1024
	default:
		return openai.ImageGenerateParamsSize1024x1024
	}
}

// --- main ---
//
// generateImage генерирует одну картинку через gpt-image-1.
// size может быть ratio ("1:1","3:2","2:3") или валидное "1024x1536".
// responseFormat: "url" | "b64_json".
func generateImage(ctx context.Context, prompt, size, responseFormat string) (string, error) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", errors.New("empty prompt")
	}
	if strings.TrimSpace(size) == "" {
		size = "1:1"
	}
	if strings.TrimSpace(responseFormat) == "" {
		responseFormat = "url"
	}

	// Нормализация ratio → пиксели
	size = ratioToSize(size)

	// Лог вызова
	start := time.Now()
	aiID, _ := aiLogStart(ctx, nil, "gpt-image-1", preview512(prompt))

	params := openai.ImageGenerateParams{
		Model:  "gpt-image-1",
		Prompt: prompt,
		Size:   toImgSize(size),
	}

	resp, err := aiClient.Images.Generate(ctx, params)
	if err != nil {
		aiLogFinish(ctx, aiID, "", err.Error(), nil, time.Since(start))
		return "", err
	}
	aiLogFinish(ctx, aiID, "", "", nil, time.Since(start))

	if len(resp.Data) == 0 {
		return "", errors.New("empty image response: no data")
	}

	b64 := strings.TrimSpace(resp.Data[0].B64JSON)
	url := strings.TrimSpace(resp.Data[0].URL)

	if strings.ToLower(responseFormat) == "b64_json" {
		if b64 == "" {
			return "", errors.New("image API returned empty b64_json")
		}
		return b64, nil
	}

	if url != "" {
		return url, nil
	}
	if b64 != "" {
		return "data:image/png;base64," + b64, nil
	}

	return "", errors.New("image API returned neither url nor b64_json")
}
