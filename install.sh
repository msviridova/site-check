#!/bin/bash

# Простой скрипт установки Site Check без клонирования репозитория
# Использование: bash install.sh

set -e

echo "🚀 Site Check - Быстрая установка"
echo "=================================="

# Проверяем права root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Запустите скрипт от имени root: sudo bash install.sh"
    exit 1
fi

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo "📦 Обновляем систему и устанавливаем пакеты..."
apt update && apt upgrade -y
apt install -y curl wget git nginx certbot python3-certbot-nginx ufw
print_status "Базовые пакеты установлены"

echo "🐹 Устанавливаем Go..."
if ! command -v go &> /dev/null; then
    cd /tmp
    wget -q https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
    tar -xzf go1.21.5.linux-amd64.tar.gz
    mv go /usr/local/
    echo 'export PATH=$PATH:/usr/local/go/bin' >> /etc/profile
    export PATH=$PATH:/usr/local/go/bin
    print_status "Go установлен: $(go version)"
else
    print_status "Go уже установлен: $(go version)"
fi

echo "👤 Создаем пользователя sitecheck..."
useradd -r -s /bin/false -d /opt/sitecheck sitecheck 2>/dev/null || true
mkdir -p /opt/sitecheck /var/log/sitecheck
chown sitecheck:sitecheck /opt/sitecheck /var/log/sitecheck
print_status "Пользователь и директории созданы"

echo "🔑 Настраиваем SSH-ключ для GitHub..."
mkdir -p /root/.ssh && chmod 700 /root/.ssh

if [ ! -f "/root/.ssh/id_ed25519" ]; then
    ssh-keygen -t ed25519 -C "server@adssitecheck.gocpa.ru" -f /root/.ssh/id_ed25519 -N ""
    print_status "SSH-ключ сгенерирован"
else
    print_status "SSH-ключ уже существует"
fi

ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

echo ""
echo "🔑 ВАЖНО: Добавьте этот SSH-ключ в GitHub:"
echo "=========================================="
cat /root/.ssh/id_ed25519.pub
echo "=========================================="
echo ""
echo "Инструкция:"
echo "1. Перейдите на https://github.com/settings/ssh/new"
echo "2. Скопируйте и вставьте ключ выше в поле 'Key'"
echo "3. Дайте ему название (например: 'Production Server adssitecheck.gocpa.ru')"
echo "4. Нажмите 'Add SSH key'"
echo ""
read -p "Нажмите Enter после добавления SSH-ключа в GitHub..."

# Тестируем подключение
echo "🔍 Тестируем SSH подключение к GitHub..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    print_status "SSH подключение к GitHub работает"
else
    print_warning "Не удалось подтвердить SSH подключение"
    echo "Продолжаем установку..."
fi

echo "📥 Клонируем репозиторий..."
cd /opt/sitecheck
if [ -d ".git" ]; then
    git pull origin master
    print_status "Репозиторий обновлен"
else
    if git clone git@github.com:gocpa/ads-site-check.git .; then
        print_status "Репозиторий склонирован"
    else
        print_error "Ошибка клонирования репозитория"
        echo "Проверьте:"
        echo "1. SSH-ключ добавлен в GitHub"
        echo "2. У вас есть доступ к репозиторию gocpa/ads-site-check"
        echo "3. SSH подключение к GitHub работает: ssh -T git@github.com"
        exit 1
    fi
fi

echo "🔨 Собираем приложение..."
go mod tidy
go build -o sitecheck main.go
chown sitecheck:sitecheck sitecheck
chmod +x sitecheck
print_status "Приложение собрано"

echo "⚙️ Создаем systemd сервис..."
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

echo "🔒 Настраиваем файрвол..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 'Nginx Full'
ufw --force enable
print_status "Файрвол настроен"

echo "🚀 Запускаем сервисы..."
systemctl daemon-reload
systemctl enable sitecheck
systemctl start sitecheck
systemctl enable nginx
systemctl restart nginx
print_status "Сервисы запущены"

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
echo "1. Убедитесь, что DNS запись adssitecheck.gocpa.ru указывает на этот сервер (5.129.234.157)"
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
print_status "Site Check готов к работе!"
