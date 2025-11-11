# Анализ рефакторинга кодовой базы

## 🔴 Критические проблемы

### 1. Глобальное состояние (Global State)
**Проблема:** Использование глобальных переменных делает код не тестируемым и создаёт скрытые зависимости.

**Места:**
- `db *sql.DB` в `db.go:16`
- `aiClient openai.Client` в `ai.go:7`
- `useAI`, `modelName`, `httpClient` в `main.go:16-24`
- `promptCache` в `ai_creatives.go:19-22`

**Решение:** Внедрить dependency injection через структуру сервиса/контекста.

```go
type App struct {
    DB       *sql.DB
    AIClient openai.Client
    Config   *Config
    // ...
}
```

---

### 2. Дублирование кода AI-вызовов
**Проблема:** Одинаковая логика вызова AI повторяется в 6+ местах.

**Места:**
- `handler.go:97-104` - classifyHandler
- `ai_creatives.go:73-80` - generateAllTextCreatives
- `ai_creatives.go:116-123` - generateKeywords
- `ai_creatives.go:173-180` - generateNegatives
- `ai_creatives.go:230-237` - generateAds
- `ai_creatives.go:303-310` - generateGraphic

**Решение:** Создать общую функцию `callAICompletion(ctx, prompt, params)`.

```go
type AIParams struct {
    MaxTokens   int
    Temperature float64
    Model       string
}

func (app *App) callAICompletion(ctx context.Context, prompt string, params AIParams) (string, error) {
    // единая логика вызова
}
```

---

### 3. Ручная подстановка плейсхолдеров
**Проблема:** Множественные `strings.ReplaceAll` вместо использования шаблонов.

**Места:**
- `handler.go:84-91` - 5 вызовов ReplaceAll
- `ai_creatives.go:287-293` - 7 вызовов ReplaceAll
- `ai_creatives.go:71,114,171,228` - повторяющиеся паттерны

**Решение:** Использовать `text/template` или создать функцию `replacePlaceholders`.

```go
func replacePlaceholders(template string, data map[string]string) string {
    // используем text/template или простую замену через map
}
```

---

### 4. Игнорирование ошибок логирования
**Проблема:** Ошибки логирования игнорируются, что может скрыть проблемы.

**Места:**
- `handler.go:44` - `apiID, _ := apiLogStart(...)`
- `handler.go:95` - `aiID, _ := aiLogStart(...)`
- `creative_handler.go:53` - `apiID, _ := apiLogStart(...)`

**Решение:** Логировать ошибки логирования или использовать отдельный канал для критических ошибок.

```go
apiID, err := apiLogStart(...)
if err != nil {
    log.Printf("WARN: failed to log API start: %v", err)
}
```

---

### 5. Длинные функции-обработчики
**Проблема:** Handlers содержат слишком много логики (валидация, бизнес-логика, логирование, ответ).

**Места:**
- `classifyHandler` - 188 строк
- `creativeHandler` - 135 строк
- `imageHandler` - 127 строк

**Решение:** Разделить на слои:
- Валидация запроса
- Бизнес-логика (отдельный сервис)
- Формирование ответа
- Логирование (middleware)

---

## 🟡 Важные улучшения

### 6. Прямой доступ к БД
**Проблема:** Нет абстракции для доступа к данным, сложно мокировать для тестов.

**Места:**
- Все функции в `db.go` напрямую работают с `*sql.DB`
- Глобальный `db` используется везде

**Решение:** Создать repository pattern.

```go
type PromptRepository interface {
    GetPrompt(key, locale string, version int) (*Prompt, error)
    GetAllPrompts() ([]Prompt, error)
}

type promptRepo struct {
    db *sql.DB
}
```

---

### 7. Жёстко заданные константы
**Проблема:** Магические числа и строки разбросаны по коду.

**Места:**
- `handler.go:61` - `12000` (лимит текста)
- `handler.go:38` - `45*time.Second` (таймаут)
- `ai_creatives.go:264` - `8000` (maxSiteText)
- `ai_creatives.go:265` - `120*time.Second` (callTimeout)
- `creative_handler.go:71` - `10000` (лимит текста)

**Решение:** Вынести в конфигурационную структуру.

```go
type Config struct {
    SiteTextMaxLength int
    ClassifyTimeout   time.Duration
    CreativeTimeout   time.Duration
    AITimeout         time.Duration
}
```

---

### 8. Несогласованная обработка ошибок
**Проблема:** Разные стили обработки ошибок в разных местах.

**Примеры:**
- Иногда `http.Error`, иногда `writeJSONError`
- Иногда логируется, иногда нет
- Разные форматы сообщений об ошибках

**Решение:** Создать единый обработчик ошибок.

```go
type APIError struct {
    Code    int
    Message string
    Err     error
}

func handleError(w http.ResponseWriter, err APIError) {
    // единая логика
}
```

---

### 9. Дублирование парсинга JSON ответов AI
**Проблема:** Похожая логика парсинга в `generateKeywords`, `generateNegatives`.

**Места:**
- `ai_creatives.go:133-159` - generateKeywords
- `ai_creatives.go:190-216` - generateNegatives

**Решение:** Создать общую функцию парсинга с fallback-стратегиями.

```go
func parseAIStringArrayResponse(raw string) ([]string, error) {
    // пробуем разные форматы
}
```

---

### 10. Отсутствие middleware для логирования
**Проблема:** Логирование API разбросано по всем handlers.

**Места:**
- Каждый handler вызывает `apiLogStart` и `apiLogFinish` вручную

**Решение:** Создать middleware для автоматического логирования.

```go
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // автоматическое логирование
    })
}
```

---

## 🟢 Улучшения качества кода

### 11. Неиспользуемый код
**Проблема:** Есть функции, которые не используются.

**Места:**
- `creative_image_handler.go:131-198` - `materializeImageToStatic`, `downloadToStatic`, `openInBrowser` и др.
- `creative_image_handler.go:23-34` - `imageRequest`, `imageResponse` (дублируют типы из `types.go`)

**Решение:** Удалить неиспользуемый код или пометить как deprecated.

---

### 12. Недостаточная валидация входных данных
**Проблема:** Валидация выполняется частично или поздно.

**Места:**
- `creative_handler.go:71` - валидация `siteText` после fetch
- `image_handler.go:90` - слабая валидация `concept`

**Решение:** Создать функции валидации для каждого типа запроса.

```go
func validateCreativeRequest(req CreativeRequest) error {
    // все проверки в одном месте
}
```

---

### 13. Смешение concerns в функциях
**Проблема:** Функции делают слишком много разных вещей.

**Примеры:**
- `fetchHTML` в `fetch.go:42` - делает и fetch, и fallback
- `extractVisibleText` в `fetch.go:92` - парсит HTML и фильтрует текст

**Решение:** Разделить на более мелкие, сфокусированные функции.

---

### 14. Неоптимальное использование кэша промптов
**Проблема:** Кэш промптов не имеет TTL и никогда не обновляется.

**Места:**
- `ai_creatives.go:19-51` - кэш без expiration

**Решение:** Добавить TTL или механизм инвалидации.

```go
type cachedPrompt struct {
    prompt  *Prompt
    expires time.Time
}
```

---

### 15. Отсутствие структурированного логирования
**Проблема:** Используется стандартный `log`, нет структурированных логов.

**Решение:** Внедрить структурированное логирование (например, `zerolog` или `zap`).

```go
logger.Info().
    Str("route", route).
    Str("url", url).
    Msg("API request started")
```

---

### 16. Хардкод путей и настроек
**Проблема:** Пути к файлам и настройки захардкожены.

**Места:**
- `creative_image_handler.go:142` - `"static"` захардкожен
- `main.go:71` - `:8080` захардкожен
- `main.go:21` - `"gpt-4.1"` захардкожен

**Решение:** Вынести в конфигурацию или environment variables.

---

### 17. Дублирование типов ответов
**Проблема:** Есть `CreativeResponse` в `types.go` и `creativeResponse` в `creative_handler.go`.

**Места:**
- `types.go:24` - `CreativeResponse`
- `creative_handler.go:13` - `creativeResponse`

**Решение:** Использовать один тип.

---

### 18. Отсутствие контекста в некоторых функциях
**Проблема:** Некоторые функции не принимают `context.Context`.

**Места:**
- `extractBrand`, `extractColorsHex`, `deriveStyleNotes` - не принимают context

**Решение:** Добавить context везде, где возможны долгие операции.

---

### 19. Неэффективная сортировка
**Проблема:** Используется пузырьковая сортировка вместо встроенной.

**Места:**
- `analyze.go:106-112` - ручная сортировка массива

**Решение:** Использовать `sort.Slice`.

```go
sort.Slice(arr, func(i, j int) bool {
    return arr[j].v > arr[i].v
})
```

---

### 20. Небезопасное использование строк
**Проблема:** Использование `strings.Title` (deprecated в Go 1.18+).

**Места:**
- `analyze.go:52` - `strings.Title(seg[len(seg)-2])`

**Решение:** Использовать `golang.org/x/text/cases`.

---

## 📋 Приоритеты рефакторинга

### Высокий приоритет (делать первым):
1. ✅ Убрать глобальное состояние (DI)
2. ✅ Вынести общую логику AI-вызовов
3. ✅ Разделить длинные handlers
4. ✅ Унифицировать обработку ошибок

### Средний приоритет:
5. ✅ Создать repository pattern для БД
6. ✅ Вынести константы в конфигурацию
7. ✅ Создать middleware для логирования
8. ✅ Использовать шаблоны для плейсхолдеров

### Низкий приоритет:
9. ✅ Удалить неиспользуемый код
10. ✅ Улучшить валидацию
11. ✅ Добавить структурированное логирование
12. ✅ Оптимизировать кэш промптов

---

## 🎯 Рекомендуемый план действий

1. **Фаза 1: Структура и DI**
   - Создать структуру `App` с зависимостями
   - Внедрить DI в handlers
   - Убрать глобальные переменные

2. **Фаза 2: Унификация логики**
   - Создать общую функцию AI-вызовов
   - Создать функцию подстановки плейсхолдеров
   - Унифицировать обработку ошибок

3. **Фаза 3: Разделение ответственности**
   - Разделить handlers на слои
   - Создать repository pattern
   - Добавить middleware

4. **Фаза 4: Улучшение качества**
   - Вынести конфигурацию
   - Добавить валидацию
   - Улучшить логирование
   - Удалить неиспользуемый код

