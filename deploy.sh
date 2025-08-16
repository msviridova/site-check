#!/bin/bash

# Скрипт автоматической установки Site Check на Linux сервер
# Использование: bash deploy.sh

set -e

echo "🚀 Начинаем установку Site Check..."

# Проверяем, что скрипт запущен от root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Запустите скрипт от имени root: sudo bash deploy.sh"
    exit 1
fi

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Обновление системы
echo "📦 Обновляем систему..."
apt update && apt upgrade -y
print_status "Система обновлена"

# 2. Установка базовых пакетов
echo "📦 Устанавливаем базовые пакеты..."
apt install -y curl wget git nginx certbot python3-certbot-nginx ufw
print_status "Базовые пакеты установлены"

# 3. Установка Go
echo "🐹 Устанавливаем Go..."
if ! command -v go &> /dev/null; then
    cd /tmp
    wget -q https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
    tar -xzf go1.21.5.linux-amd64.tar.gz
    mv go /usr/local/
    
    echo 'export PATH=$PATH:/usr/local/go/bin' >> /etc/profile
    echo 'export GOPATH=/opt/go' >> /etc/profile
    echo 'export GOBIN=/opt/go/bin' >> /etc/profile
    
    export PATH=$PATH:/usr/local/go/bin
    export GOPATH=/opt/go
    export GOBIN=/opt/go/bin
    
    print_status "Go установлен: $(go version)"
else
    print_status "Go уже установлен: $(go version)"
fi

# 4. Создание пользователя
echo "👤 Создаем пользователя sitecheck..."
if ! id "sitecheck" &>/dev/null; then
    useradd -r -s /bin/false -d /opt/sitecheck sitecheck
    print_status "Пользователь sitecheck создан"
else
    print_status "Пользователь sitecheck уже существует"
fi

# 5. Создание директорий
echo "📁 Создаем директории..."
mkdir -p /opt/sitecheck
mkdir -p /var/log/sitecheck
chown sitecheck:sitecheck /opt/sitecheck /var/log/sitecheck
print_status "Директории созданы"

# 6. Клонирование и сборка
echo "📥 Клонируем репозиторий..."
cd /opt/sitecheck
if [ -d ".git" ]; then
    git pull origin master
    print_status "Репозиторий обновлен"
else
    git clone https://github.com/gocpa/ads-site-check.git .
    print_status "Репозиторий склонирован"
fi

echo "🔨 Собираем приложение..."
go mod tidy
go build -o sitecheck main.go
chown sitecheck:sitecheck sitecheck
chmod +x sitecheck
print_status "Приложение собрано"

# 7. Создание systemd сервиса
echo "⚙️  Создаем systemd сервис..."
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
print_status "Systemd сервис создан"

# 8. Настройка Nginx
echo "🌐 Настраиваем Nginx..."
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

ln -sf /etc/nginx/sites-available/adssitecheck.gocpa.ru /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

if nginx -t; then
    print_status "Nginx сконфигурирован"
else
    print_error "Ошибка в конфигурации Nginx"
    exit 1
fi

# 9. Настройка файрвола
echo "🔒 Настраиваем файрвол..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 'Nginx Full'
ufw --force enable
print_status "Файрвол настроен"

# 10. Запуск сервисов
echo "🚀 Запускаем сервисы..."
systemctl daemon-reload
systemctl enable sitecheck
systemctl start sitecheck
systemctl enable nginx
systemctl restart nginx
print_status "Сервисы запущены"

# 11. Проверка работоспособности
echo "🔍 Проверяем работоспособность..."
sleep 3

if systemctl is-active --quiet sitecheck; then
    print_status "Сервис sitecheck работает"
else
    print_error "Сервис sitecheck не запущен"
    echo "Логи сервиса:"
    journalctl -u sitecheck --no-pager -n 10
fi

if systemctl is-active --quiet nginx; then
    print_status "Nginx работает"
else
    print_error "Nginx не запущен"
fi

# Проверяем health check
if curl -s http://localhost:8080/healthz > /dev/null; then
    print_status "Health check работает"
else
    print_warning "Health check не отвечает"
fi

echo ""
echo "🎉 Установка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Убедитесь, что DNS запись adssitecheck.gocpa.ru указывает на этот сервер"
echo "2. Получите SSL сертификат:"
echo "   certbot --nginx -d adssitecheck.gocpa.ru --non-interactive --agree-tos --email admin@gocpa.ru"
echo ""
echo "🔧 Полезные команды:"
echo "   Логи приложения: tail -f /var/log/sitecheck/sitecheck.log"
echo "   Статус сервиса:  systemctl status sitecheck"
echo "   Перезапуск:      systemctl restart sitecheck"
echo ""
echo "🌐 Тестирование:"
echo "   Health check: curl http://localhost:8080/healthz"
echo "   API test:     curl -X POST http://localhost:8080/classify -H 'Content-Type: application/json' -d '{\"url\": \"https://google.com\"}'"
echo ""
print_status "Готово к работе!"
