# Быстрый старт Site Check

## 🚀 Автоматическая установка (рекомендуется)

1. **Подключитесь к серверу:**
   ```bash
   ssh root@5.129.234.157
   ```

2. **Скачайте и запустите скрипт установки:**
   ```bash
   # Способ 1: Через клонирование репозитория (если есть доступ)
   git clone https://github.com/gocpa/ads-site-check.git
   cd ads-site-check
   chmod +x install.sh
   ./install.sh
   
   # Способ 2: Создание скрипта вручную (если репозиторий приватный)
   # Скопируйте содержимое install.sh из GitHub и выполните:
   nano install.sh
   # Вставьте код скрипта, сохраните (Ctrl+X, Y, Enter)
   chmod +x install.sh
   ./install.sh
   
   # Способ 3: Встроенная установка (одной командой)
   bash <(cat << 'DEPLOY_SCRIPT'
#!/bin/bash
set -e
echo "🚀 Начинаем установку Site Check..."
if [ "$EUID" -ne 0 ]; then
    echo "❌ Запустите от имени root: sudo bash"
    exit 1
fi
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo "📦 Обновляем систему..."
apt update && apt upgrade -y
apt install -y curl wget git nginx certbot python3-certbot-nginx ufw

echo "🐹 Устанавливаем Go..."
if ! command -v go &> /dev/null; then
    cd /tmp
    wget -q https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
    tar -xzf go1.21.5.linux-amd64.tar.gz
    mv go /usr/local/
    echo 'export PATH=$PATH:/usr/local/go/bin' >> /etc/profile
    export PATH=$PATH:/usr/local/go/bin
fi

echo "👤 Создаем пользователя..."
useradd -r -s /bin/false -d /opt/sitecheck sitecheck 2>/dev/null || true
mkdir -p /opt/sitecheck /var/log/sitecheck
chown sitecheck:sitecheck /opt/sitecheck /var/log/sitecheck

echo "🔑 Настраиваем SSH для GitHub..."
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ ! -f "/root/.ssh/id_ed25519" ]; then
    ssh-keygen -t ed25519 -C "server@adssitecheck.gocpa.ru" -f /root/.ssh/id_ed25519 -N ""
fi
ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

echo "🔑 Добавьте этот SSH-ключ в GitHub:"
echo "=========================================="
cat /root/.ssh/id_ed25519.pub
echo "=========================================="
echo "1. Перейдите: https://github.com/settings/ssh/new"
echo "2. Вставьте ключ выше"
echo "3. Нажмите 'Add SSH key'"
read -p "Нажмите Enter после добавления ключа..."

echo "📥 Клонируем репозиторий..."
cd /opt/sitecheck
git clone git@github.com:gocpa/ads-site-check.git . || {
    echo "❌ Ошибка клонирования. Проверьте SSH-ключ в GitHub"
    exit 1
}

echo "🔨 Собираем приложение..."
go mod tidy && go build -o sitecheck main.go
chown sitecheck:sitecheck sitecheck && chmod +x sitecheck

echo "⚙️ Создаем сервис..."
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
Environment=USE_AI=false
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/sitecheck
[Install]
WantedBy=multi-user.target
EOF

echo "🌐 Настраиваем Nginx..."
cat > /etc/nginx/sites-available/adssitecheck.gocpa.ru << 'EOF'
server {
    listen 80;
    server_name adssitecheck.gocpa.ru;
    access_log /var/log/nginx/adssitecheck.gocpa.ru.access.log;
    error_log /var/log/nginx/adssitecheck.gocpa.ru.error.log;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    location /healthz {
        proxy_pass http://127.0.0.1:8080/healthz;
        access_log off;
    }
}
EOF

ln -sf /etc/nginx/sites-available/adssitecheck.gocpa.ru /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t || exit 1

echo "🔒 Настраиваем файрвол..."
ufw --force reset
ufw default deny incoming && ufw default allow outgoing
ufw allow ssh && ufw allow 'Nginx Full' && ufw --force enable

echo "🚀 Запускаем сервисы..."
systemctl daemon-reload
systemctl enable sitecheck && systemctl start sitecheck
systemctl enable nginx && systemctl restart nginx

echo "🎉 Установка завершена!"
echo "Следующие шаги:"
echo "1. Настройте DNS: adssitecheck.gocpa.ru -> 5.129.234.157"
echo "2. Получите SSL: certbot --nginx -d adssitecheck.gocpa.ru --agree-tos --email admin@gocpa.ru"
echo "3. Тест: curl http://localhost:8080/healthz"
DEPLOY_SCRIPT
)
   ```

3. **Добавьте SSH-ключ в GitHub:**
   - Скрипт покажет SSH-ключ, который нужно добавить
   - Перейдите на https://github.com/settings/ssh/new
   - Вставьте ключ и сохраните
   - Нажмите Enter в терминале для продолжения

4. **Настройте DNS:**
   - Убедитесь, что `adssitecheck.gocpa.ru` указывает на `5.129.234.157`

5. **Получите SSL сертификат:**
   ```bash
   certbot --nginx -d adssitecheck.gocpa.ru --non-interactive --agree-tos --email admin@gocpa.ru
   ```

## ✅ Проверка работы

```bash
# Health check
curl https://adssitecheck.gocpa.ru/healthz

# Тест API
curl -X POST https://adssitecheck.gocpa.ru/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://google.com"}'
```

## 📋 Полная документация

Подробная инструкция находится в файле [DEPLOYMENT.md](DEPLOYMENT.md)

## 🔧 Управление сервисом

```bash
# Статус
systemctl status sitecheck

# Перезапуск
systemctl restart sitecheck

# Логи
tail -f /var/log/sitecheck/sitecheck.log
```

## 🆘 Поддержка

При возникновении проблем:
1. Проверьте логи: `journalctl -u sitecheck -f`
2. Убедитесь в правильности DNS настроек
3. Проверьте статус сервисов: `systemctl status sitecheck nginx`
