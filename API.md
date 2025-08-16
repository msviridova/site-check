# Site Check API Documentation

## Описание

Site Check API предоставляет сервис для анализа и классификации веб-сайтов. API принимает URL сайта и возвращает краткое описание его тематики на русском языке.

## Базовая информация

- **Протокол:** HTTP/HTTPS
- **Формат данных:** JSON
- **Кодировка:** UTF-8
- **Порт по умолчанию:** 8080

## Endpoints

### 1. Анализ сайта

Основной endpoint для анализа содержимого веб-сайта.

**URL:** `/classify`  
**Метод:** `POST`  
**Content-Type:** `application/json`

#### Запрос

**Параметры:**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `url` | string | Да | URL сайта для анализа |

**Пример запроса:**
```json
{
  "url": "https://example.com"
}
```

**Требования к URL:**
- Должен быть валидным URL
- Должен содержать схему (http:// или https://)
- Должен содержать домен
- Максимальная длина: не ограничена
- Поддерживаемые протоколы: HTTP, HTTPS

#### Ответ

**Успешный ответ (200 OK):**

| Поле | Тип | Описание |
|------|-----|----------|
| `summary` | string | Краткое описание тематики сайта на русском языке |
| `lang` | string | Язык ответа (всегда "ru") |
| `source` | string | Источник анализа: "ai" или "heuristic" |

**Пример ответа:**
```json
{
  "summary": "Интернет-магазин электроники и бытовой техники",
  "lang": "ru",
  "source": "ai"
}
```

**Возможные значения `source`:**
- `"ai"` - анализ выполнен с использованием OpenAI API
- `"heuristic"` - анализ выполнен эвристическими методами

#### Коды ошибок

| Код | Описание | Пример ответа |
|-----|----------|---------------|
| 400 | Неверный запрос | `"bad JSON"` |
| 400 | URL не указан | `"url is required"` |
| 400 | Неверный URL | `"invalid url"` |
| 405 | Неверный HTTP метод | `"use POST"` |
| 502 | Ошибка загрузки сайта | `"fetch failed: ..."` |

### 2. Health Check

Endpoint для проверки состояния сервиса.

**URL:** `/healthz`  
**Метод:** `GET`

#### Ответ

**Успешный ответ (200 OK):**
```
ok
```

## Примеры использования

### cURL

```bash
# Анализ сайта
curl -X POST https://your-domain.com/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Health check
curl https://your-domain.com/healthz
```

### JavaScript (fetch)

```javascript
// Анализ сайта
async function analyzeSite(url) {
  try {
    const response = await fetch('https://your-domain.com/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Использование
analyzeSite('https://example.com')
  .then(result => {
    console.log('Анализ сайта:', result.summary);
    console.log('Источник:', result.source);
  })
  .catch(error => {
    console.error('Ошибка анализа:', error);
  });
```

### Python (requests)

```python
import requests
import json

def analyze_site(url):
    """Анализ сайта через Site Check API"""
    api_url = "https://your-domain.com/classify"
    
    payload = {"url": url}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(api_url, 
                               data=json.dumps(payload), 
                               headers=headers,
                               timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

# Использование
result = analyze_site("https://example.com")
if result:
    print(f"Описание: {result['summary']}")
    print(f"Источник анализа: {result['source']}")
```

### PHP

```php
<?php
function analyzeSite($url) {
    $apiUrl = 'https://your-domain.com/classify';
    
    $data = json_encode(['url' => $url]);
    
    $options = [
        'http' => [
            'header' => "Content-Type: application/json\r\n",
            'method' => 'POST',
            'content' => $data,
            'timeout' => 30
        ]
    ];
    
    $context = stream_context_create($options);
    $result = file_get_contents($apiUrl, false, $context);
    
    if ($result === FALSE) {
        return null;
    }
    
    return json_decode($result, true);
}

// Использование
$result = analyzeSite('https://example.com');
if ($result) {
    echo "Описание: " . $result['summary'] . "\n";
    echo "Источник: " . $result['source'] . "\n";
}
?>
```

## Особенности работы

### Таймауты

- **Общий таймаут запроса:** 12 секунд
- **Таймаут загрузки сайта:** 10 секунд
- **Таймаут AI-анализа:** 15 секунд (при включенном AI)

### Лимиты

- **Размер HTML:** максимум 2MB
- **Размер текста для AI:** максимум 4000 символов
- **Длина ответа:** обычно 50-200 символов

### Обработка контента

1. **Загрузка HTML:** Сервис загружает HTML страницы по указанному URL
2. **Извлечение текста:** Удаляются теги script, style, nav и другие нерелевантные элементы
3. **Анализ содержимого:** 
   - При включенном AI: отправка в OpenAI для анализа
   - При отключенном AI: эвристический анализ по ключевым словам
4. **Формирование ответа:** Возврат краткого описания на русском языке

### Fallback механизмы

Если основной анализ не удается, сервис использует:
1. Title страницы
2. Meta description
3. Доменное имя
4. Стандартное сообщение "Не удалось определить тематику сайта"

## Коды состояния HTTP

| Код | Статус | Описание |
|-----|--------|----------|
| 200 | OK | Успешный анализ |
| 400 | Bad Request | Ошибка в запросе (неверный JSON, отсутствует URL, неверный URL) |
| 405 | Method Not Allowed | Использован неверный HTTP метод |
| 502 | Bad Gateway | Ошибка при загрузке целевого сайта |
| 500 | Internal Server Error | Внутренняя ошибка сервера |

## Рекомендации по использованию

### Обработка ошибок

Всегда обрабатывайте возможные ошибки:
- Проверяйте HTTP статус код
- Обрабатывайте таймауты
- Предусмотрите fallback для недоступности сервиса

### Оптимизация запросов

- Используйте connection pooling для множественных запросов
- Кэшируйте результаты для часто запрашиваемых URL
- Устанавливайте разумные таймауты на стороне клиента

### Безопасность

- Валидируйте URL на стороне клиента
- Не передавайте чувствительные данные в URL
- Используйте HTTPS для продакшн окружения

## Мониторинг и отладка

### Health Check

Регулярно проверяйте доступность сервиса через `/healthz` endpoint:

```bash
# Простая проверка доступности
curl -f https://your-domain.com/healthz

# Проверка с таймаутом
curl --max-time 5 https://your-domain.com/healthz
```

### Логирование

Сервис логирует:
- Все входящие запросы с URL
- Ошибки загрузки сайтов
- Время выполнения запросов
- Использование AI vs эвристического анализа

### Метрики производительности

Типичное время ответа:
- **Эвристический анализ:** 1-3 секунды
- **AI-анализ:** 3-8 секунд
- **Health check:** < 100ms

## Поддержка

При возникновении проблем проверьте:
1. Корректность формата запроса
2. Доступность целевого URL
3. Статус сервиса через health check
4. Логи сервера для диагностики ошибок
