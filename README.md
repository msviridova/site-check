# Site Check Service

## Database Setup

```sql
CREATE DATABASE sitecheck CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'sitecheck'@'localhost' IDENTIFIED BY '1234509876';
GRANT ALL PRIVILEGES ON sitecheck.* TO 'sitecheck'@'localhost';
FLUSH PRIVILEGES;

-- api_logs
CREATE TABLE IF NOT EXISTS api_logs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  url TEXT,
  request_body LONGTEXT,
  response_body LONGTEXT,
  error_text TEXT,
  status_code INT,
  duration_ms INT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ai_logs
CREATE TABLE IF NOT EXISTS ai_logs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  api_log_id BIGINT,
  model VARCHAR(64),
  prompt_preview TEXT,
  response_body LONGTEXT,
  error_text TEXT,
  prompt_tokens INT,
  completion_tokens INT,
  total_tokens INT,
  duration_ms INT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY (api_log_id),
  CONSTRAINT fk_ai_api FOREIGN KEY (api_log_id) REFERENCES api_logs(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## Prompt Templates (seed data for `prompts` table)

Все ключи хранятся в таблице `prompts`. Ниже — рекомендуемые тексты шаблонов, которые можно положить в базу. Плейсхолдеры в фигурных скобках заменяются кодом.

### `classify`
```
Ты — маркетинговый аналитик. На основе HTML страницы определяешь бренд, стилистику и цветовую палитру.

Данные для анализа:
- эвристический бренд: {HEUR_BRAND}
- эвристические заметки о стиле: {HEUR_STYLE}
- эвристические цвета: [{HEUR_COLORS}]
- видимый текст сайта:
{SITE_TEXT}

Требования:
- Верни СТРОГО JSON со структурами:
{
  "summary": "...",
  "brand": "...",
  "style_notes": "...",
  "main_colors_hex": ["#RRGGBB", ...],
  "additional_colors_hex": ["#RRGGBB", ...],
  "background_color_hex": "#RRGGBB",
  "accent_primary_hex": "#RRGGBB",
  "accent_secondary_hex": "#RRGGBB"
}
- Цвета только в HEX-формате.
- Если данных не хватает — делай аккуратные допущения и упоминай их в summary.
```

### `creative_text_keywords`
```
Генерация ключевых слов
Ты — ИИ-агент, задача которого — сформировать максимально релевантный и эффективный список ключевых слов и фраз для запуска контекстной рекламы в Яндекс.Директ.
Анализируй только предоставленный текст сайта.

Контент сайта:
{website_text}

Требования по формату ответа:
Верни СТРОГО валидный JSON вида:
{
  "keywords": {
    "общие": [],
    "продукты_услуги": [],
    "бренды": [],
    "уточняющие": []
  }
}

Генерируй 30–40 запросов, без «дешево/лучший/топ», без орфографических ошибок, только релевантные.
```

### `creative_text_negatives`
```
Генерация минус-слов
Ты — ИИ-агент, задача которого — помочь маркетологу составить список минус-слов для Яндекс.Директ.
Анализируй только предоставленный текст.

Контент сайта:
{website_text}

Формат ответа — СТРОГО JSON:
{
  "negatives": {
    "общие": [],
    "бесплатные": [],
    "негатив_и_инфо": [],
    "вакансии": [],
    "конкуренты": [],
    "нерелевант": []
  }
}
```

### `creative_text_ads`
```
Ты — ИИ-агент для генерации рекламных объявлений Яндекс.Директ.
Анализируй текст сайта:

{website_text}

Сформируй СТРОГО JSON-массив из 5 объявлений:
[
  {
    "id": "AD1",
    "header": "Заголовок ≤56 символов",
    "text": "Текст ≤81 символ, с CTA",
    "links": [
      {"url":"https://...", "title":"≤30", "desc":"≤60"},
      {"url":"https://...", "title":"≤30", "desc":"≤60"}
    ],
    "details": ["Уточнение1", "Уточнение2"]
  },
  ...
]
Соблюдай лимиты. Если данных мало — пропускай поле.
```

### `creative_text_all`
```
Ты — ИИ-ассистент для маркетолога. На основе текста сайта нужно подготовить набор материалов для рекламной кампании.

Контент сайта:
{website_text}

Верни СТРОГО JSON вида:
{
  "keywords": ["..."],
  "negatives": ["..."],
  "ads": [
    {
      "id": "AD1",
      "header": "...",
      "text": "...",
      "cta": "..."
    }
  ]
}

Требования:
- keywords — только релевантные поисковые фразы (30–40 штук).
- negatives — минус-слова для фильтрации нецелевых запросов.
- ads — минимум 3 объявления (headline ≤56 символов, text ≤81 символ, добавь короткий CTA).
- Ответ должен быть валидным JSON без комментариев.
```

### `creative_graphic`
```
Твоя роль: арт-директор performance-рекламы.
Задача: на основе сайта сгенерировать 3 визуальных концепта баннера для РСЯ и выдать готовые промпты для генераторов изображений (Midjourney / Stable Diffusion / DALL·E) + короткий текст (headline/body/cta).

site_url: {site_url}
goal: {goal}
audience: {audience}
geo: {geo}
offer_constraints: {offer_constraints}
brand_overrides: {brand_overrides}

Если нет доступа к сайту — используй контент ниже. Если что-то предположил — укажи в "assumptions".

Контент сайта:
{website_text}

Формат ответа — СТРОГО JSON. Образец структуры:
{
  "brand": {
    "name": "IKEA",
    "extracted_colors_hex": ["#0058A3", "#FFC72C"],
    "style_notes": "минимализм, скандинавский уют",
    "assumptions": "часть информации угадана"
  },
  "concepts": [
    {
      "name": "Семья и уют",
      "rationale": "молодые семьи ищут комфортную мебель",
      "visual_plan": "светлая гостиная, диван, семья из 3 человек",
      "image_prompts": {
        "1x1": "ENGLISH prompt … --ar 1:1",
        "4x1": "ENGLISH prompt … --ar 4:1",
        "1x2": "ENGLISH prompt … --ar 1:2"
      },
      "negatives": "clutter, text, watermark, busy background",
      "generator_params": { "stylize": 100, "chaos": 10, "seed": 12345 },
      "ad_copy_ru": {
        "headline": "Мебель для семьи",
        "body": "Уют и стиль для вашего дома",
        "cta": "Выбрать"
      }
    }
  ],
  "safety_checklist": ["Нет абсолютных обещаний", "Нет логотипов реальных брендов", "Минимум текста на изображении"]
}

⚠️ Правила:
- Пиши image-prompts на АНГЛИЙСКОМ (без кавычек, без фигурных скобок).
- Длина каждого image-prompt ≤ 750 символов.
- Используй HEX цвета бренда (если нет — подбери гармоничные).
- Минимум текста на изображении.
```

### `image`
```
Ты — специализированный генератор промптов для создания изображений по техническому заданию.
У тебя уже есть концепт, составленный арт-директором, и дополнительные ограничения.

Сгенерируй СТРОГО JSON вида:
{
  "prompt": "...",
  "negatives": "..."
}

Правила:
- Используй только информацию, переданную в поле `concept`.
- Строго соблюдай ограничения из `additional`.
- Пиши основной промпт на английском языке, без кавычек и фигурных скобок.
- Не добавляй ничего от себя, кроме описания сцены и стилистики.
- В `negatives` перечисли запреты через запятую на английском.
```