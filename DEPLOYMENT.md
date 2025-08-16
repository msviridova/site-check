# Инструкция по развертыванию Site Check на Linux сервере

## Описание проекта

Site Check - это Go-сервис для анализа и классификации веб-сайтов. Сервис может работать в двух режимах:
- С использованием OpenAI API для AI-анализа
- В режиме эвристического анализа (без AI)

Сервис принимает POST-запросы с URL сайта и возвращает краткое описание тематики сайта.

## Требования к серверу

- Linux сервер (Ubuntu 20.04+ или CentOS 7+)
- Минимум 1GB RAM
- 10GB свободного места на диске
- Доступ к интернету
- Права sudo

## 1. Подключение к серверу

```bash
ssh root@5.129.234.157
```

## 2. Обновление системы и установка базовых пакетов

```bash
# Обновляем систему
apt update && apt upgrade -y

# Устанавливаем необходимые пакеты
apt install -y curl wget git nginx certbot python3-certbot-nginx ufw
```

## 3. Установка Go

```bash
# Скачиваем и устанавливаем Go 1.21+
cd /tmp
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
tar -xzf go1.21.5.linux-amd64.tar.gz
mv go /usr/local/

# Настраиваем переменные окружения
echo 'export PATH=$PATH:/usr/local/go/bin' >> /etc/profile
echo 'export GOPATH=/opt/go' >> /etc/profile
echo 'export GOBIN=/opt/go/bin' >> /etc/profile
source /etc/profile

# Проверяем установку
go version
```

## 4. Создание пользователя для приложения

```bash
# Создаем пользователя sitecheck
useradd -r -s /bin/false -d /opt/sitecheck sitecheck

# Создаем директории
mkdir -p /opt/sitecheck
mkdir -p /var/log/sitecheck
chown sitecheck:sitecheck /opt/sitecheck /var/log/sitecheck
```

## 5. Клонирование и сборка проекта

```bash
# Переходим в рабочую директорию
cd /opt/sitecheck

# Клонируем репозиторий
git clone https://github.com/gocpa/ads-site-check.git .

# Собираем приложение
go mod tidy
go build -o sitecheck main.go

# Устанавливаем права
chown sitecheck:sitecheck sitecheck
chmod +x sitecheck
```

## 6. Создание systemd сервиса

Создаем файл сервиса:

```bash
cat > /etc/systemd/system/sitecheck.service << 'EOF'
[Unit]
Description=Site Check Service
After=network.target

[Service]
Type=simple
User=sitecheck
Group=sitecheck
WorkingDirectory=/opt/sitecheck
ExecStart=/opt/sitecheck/sitecheck
Restart=always
RestartSec=5
StandardOutput=append:/var/log/sitecheck/sitecheck.log
StandardError=append:/var/log/sitecheck/sitecheck.log

# Переменные окружения
Environment=USE_AI=false
# Environment=OPENAI_API_KEY=your_openai_api_key_here

# Безопасность
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/sitecheck

[Install]
WantedBy=multi-user.target
EOF
```

## 7. Настройка Nginx

Создаем конфигурацию Nginx:

```bash
cat > /etc/nginx/sites-available/adssitecheck.gocpa.ru << 'EOF'
server {
    listen 80;
    server_name adssitecheck.gocpa.ru;

    # Логи
    access_log /var/log/nginx/adssitecheck.gocpa.ru.access.log;
    error_log /var/log/nginx/adssitecheck.gocpa.ru.error.log;

    # Проксирование на Go-приложение
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Таймауты
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Health check endpoint
    location /healthz {
        proxy_pass http://127.0.0.1:8080/healthz;
        access_log off;
    }
}
EOF

# Активируем сайт
ln -s /etc/nginx/sites-available/adssitecheck.gocpa.ru /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Проверяем конфигурацию
nginx -t
```

## 8. Настройка DNS

**ВАЖНО:** Перед продолжением убедитесь, что DNS-запись для домена `adssitecheck.gocpa.ru` указывает на IP `5.129.234.157`.

Проверить можно командой:
```bash
nslookup adssitecheck.gocpa.ru
```

## 9. Настройка SSL сертификата

```bash
# Получаем SSL сертификат от Let's Encrypt
certbot --nginx -d adssitecheck.gocpa.ru --non-interactive --agree-tos --email admin@gocpa.ru

# Настраиваем автообновление сертификата
systemctl enable certbot.timer
systemctl start certbot.timer
```

## 10. Настройка файрвола

```bash
# Настраиваем UFW
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 'Nginx Full'
ufw --force enable
```

## 11. Запуск сервисов

```bash
# Перезагружаем systemd
systemctl daemon-reload

# Запускаем и включаем автозапуск сервисов
systemctl enable sitecheck
systemctl start sitecheck

systemctl enable nginx
systemctl restart nginx

# Проверяем статус
systemctl status sitecheck
systemctl status nginx
```

## 12. Проверка работоспособности

```bash
# Проверяем health check
curl http://localhost:8080/healthz

# Проверяем через домен
curl https://adssitecheck.gocpa.ru/healthz

# Тестируем основной функционал
curl -X POST https://adssitecheck.gocpa.ru/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://google.com"}'
```

## 13. Мониторинг и логи

```bash
# Просмотр логов приложения
tail -f /var/log/sitecheck/sitecheck.log

# Просмотр логов Nginx
tail -f /var/log/nginx/adssitecheck.gocpa.ru.access.log
tail -f /var/log/nginx/adssitecheck.gocpa.ru.error.log

# Статус сервиса
systemctl status sitecheck
```

## 14. Настройка OpenAI API (опционально)

Если хотите использовать AI-анализ:

```bash
# Редактируем сервис
systemctl edit sitecheck

# Добавляем в файл:
[Service]
Environment=USE_AI=true
Environment=OPENAI_API_KEY=your_actual_openai_api_key_here

# Перезапускаем сервис
systemctl daemon-reload
systemctl restart sitecheck
```

## 15. Обновление приложения

```bash
# Переходим в директорию проекта
cd /opt/sitecheck

# Останавливаем сервис
systemctl stop sitecheck

# Обновляем код
git pull origin master

# Пересобираем
go build -o sitecheck main.go
chown sitecheck:sitecheck sitecheck

# Запускаем сервис
systemctl start sitecheck
```

## API Документация

### Endpoint: `/classify`

**Метод:** POST  
**URL:** `https://adssitecheck.gocpa.ru/classify`

**Запрос:**
```json
{
  "url": "https://example.com"
}
```

**Ответ:**
```json
{
  "summary": "Краткое описание сайта",
  "lang": "ru",
  "source": "ai" // или "heuristic"
}
```

### Endpoint: `/healthz`

**Метод:** GET  
**URL:** `https://adssitecheck.gocpa.ru/healthz`

**Ответ:** `ok` (статус 200)

## Примеры использования

```bash
# Анализ сайта
curl -X POST https://adssitecheck.gocpa.ru/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://market.yandex.ru"}'

# Проверка здоровья сервиса
curl https://adssitecheck.gocpa.ru/healthz
```

## Устранение неполадок

### Сервис не запускается
```bash
# Проверяем логи
journalctl -u sitecheck -f

# Проверяем права доступа
ls -la /opt/sitecheck/sitecheck
```

### Nginx возвращает 502
```bash
# Проверяем, что приложение слушает порт 8080
netstat -tlnp | grep 8080

# Проверяем логи Nginx
tail -f /var/log/nginx/error.log
```

### SSL сертификат не работает
```bash
# Проверяем статус certbot
systemctl status certbot.timer

# Обновляем сертификат вручную
certbot renew --dry-run
```

## Безопасность

1. Регулярно обновляйте систему: `apt update && apt upgrade`
2. Мониторьте логи на подозрительную активность
3. Используйте сильные пароли для SSH
4. Рассмотрите настройку fail2ban для защиты от брутфорса
5. Регулярно делайте бэкапы конфигурации

## Производительность

Сервис оптимизирован для обработки запросов с таймаутами:
- HTTP клиент: 10 секунд
- Обработка запроса: 12 секунд  
- AI запрос: 15 секунд
- Лимит размера HTML: 2MB
- Лимит текста для AI: 4000 символов

При высокой нагрузке рассмотрите:
- Увеличение количества экземпляров сервиса
- Настройку load balancer
- Кэширование результатов
