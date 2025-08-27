# Развертывание Site Check сервиса

## Описание

Site Check - это Go-сервис для анализа и классификации веб-сайтов. Сервис может работать в двух режимах:
- С использованием OpenAI API для AI-анализа контента
- В режиме эвристического анализа (без внешних API)

## Системные требования

- Linux сервер (Ubuntu 20.04+, Debian 10+, CentOS 7+)
- Минимум 2GB RAM (рекомендуется 4GB)
- 20GB свободного места на диске
- Доступ к интернету
- Права root/sudo
- MySQL 8.0+ или MariaDB 10.5+

## Этапы развертывания

### 1. Подготовка системы

Обновите систему и установите базовые пакеты:

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка базовых пакетов
sudo apt install -y curl wget git nginx certbot python3-certbot-nginx ufw

# Установка MySQL (выберите один вариант)
sudo apt install -y mysql-server
# ИЛИ для MariaDB:
# sudo apt install -y mariadb-server

# Запуск и включение автозапуска MySQL
sudo systemctl start mysql
sudo systemctl enable mysql

# Базовая настройка безопасности MySQL
sudo mysql_secure_installation
```

### 2. Установка Go

Скачайте и установите Go версии 1.21 или выше:

```bash
# Переход в временную директорию
cd /tmp

# Скачивание Go (проверьте актуальную версию на https://golang.org/dl/)
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz

# Удаление старой версии Go (если есть)
sudo rm -rf /usr/local/go

# Распаковка Go
sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz

# Настройка переменных окружения
echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee -a /etc/profile
echo 'export GOPATH=/opt/go' | sudo tee -a /etc/profile
echo 'export GOBIN=/opt/go/bin' | sudo tee -a /etc/profile

# Применение переменных окружения
source /etc/profile

# Проверка установки
go version
```

### 3. Настройка базы данных

Настройте MySQL/MariaDB для работы приложения:

#### Создание базы данных и пользователя

```bash
# Подключение к MySQL как root
sudo mysql -u root -p

# В консоли MySQL выполните следующие команды:
```

```sql
CREATE DATABASE sitecheck CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'sitecheck_user'@'localhost' IDENTIFIED BY 'SecurePassword123!';
GRANT ALL PRIVILEGES ON sitecheck.* TO 'sitecheck_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

```bash
# Проверка подключения с новым пользователем
mysql -u sitecheck_user -p sitecheck
# Введите пароль: SecurePassword123!
```

#### Создание таблиц

```bash
# Подключение к базе данных sitecheck
mysql -u sitecheck_user -p sitecheck
# Введите пароль: SecurePassword123!
```

```sql
-- В консоли MySQL выполните создание таблиц:
USE sitecheck;

-- Таблица для логов API запросов
CREATE TABLE IF NOT EXISTS api_logs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  route VARCHAR(255),
  url TEXT,
  request_body LONGTEXT,
  response_body LONGTEXT,
  error_text TEXT,
  status_code INT,
  duration_ms INT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Таблица для логов AI запросов
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

#### Настройка производительности

```bash
# Создание резервной копии конфигурации
sudo cp /etc/mysql/mysql.conf.d/mysqld.cnf /etc/mysql/mysql.conf.d/mysqld.cnf.backup

# Добавление настроек производительности
sudo tee -a /etc/mysql/mysql.conf.d/mysqld.cnf << 'EOF'

# Настройки производительности для Site Check
[mysqld]
innodb_buffer_pool_size = 1G
innodb_log_file_size = 256M
max_connections = 200
query_cache_size = 64M
query_cache_type = 1
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 2
EOF

# Перезапуск MySQL для применения настроек
sudo systemctl restart mysql

# Проверка статуса
sudo systemctl status mysql
```

### 4. Создание пользователя для сервиса

```bash
# Создание системного пользователя sitecheck
sudo useradd -r -s /bin/false -d /opt/sitecheck sitecheck

# Создание рабочих директорий
sudo mkdir -p /opt/sitecheck
sudo mkdir -p /var/log/sitecheck

# Настройка прав доступа
sudo chown sitecheck:sitecheck /opt/sitecheck
sudo chown sitecheck:sitecheck /var/log/sitecheck
sudo chmod 755 /opt/sitecheck
sudo chmod 755 /var/log/sitecheck

# Проверка созданного пользователя
id sitecheck
ls -la /opt/ | grep sitecheck
ls -la /var/log/ | grep sitecheck
```

### 5. Настройка SSH для GitHub

Если репозиторий приватный, настройте SSH-аутентификацию:

```bash
# Создание SSH директории (если не существует)
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Генерация SSH-ключа ED25519
ssh-keygen -t ed25519 -C "server@domain.com" -f ~/.ssh/id_ed25519 -N ""

# Вывод публичного ключа для добавления в GitHub
echo "Добавьте этот SSH-ключ в GitHub (Settings -> SSH and GPG keys):"
cat ~/.ssh/id_ed25519.pub

# Добавление GitHub в known_hosts
ssh-keyscan github.com >> ~/.ssh/known_hosts

# Настройка прав доступа
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

echo "1. Скопируйте ключ выше"
echo "2. Перейдите на https://github.com/settings/ssh/new"
echo "3. Вставьте ключ и сохраните"
echo "4. Нажмите Enter для продолжения..."
read

# Тестирование подключения к GitHub
ssh -T git@github.com
```

### 6. Получение исходного кода

```bash
# Переход в рабочую директорию
cd /opt/sitecheck

# Клонирование репозитория (для приватного репозитория через SSH)
sudo -u sitecheck git clone git@github.com:username/site-check.git .

# ИЛИ для публичного репозитория через HTTPS:
# sudo -u sitecheck git clone https://github.com/username/site-check.git .

# Проверка клонирования
ls -la /opt/sitecheck
sudo -u sitecheck git status

# Проверка файлов проекта
ls -la /opt/sitecheck/
cat /opt/sitecheck/go.mod
```

### 7. Сборка приложения

```bash
# Переход в директорию проекта
cd /opt/sitecheck

# Загрузка зависимостей
sudo -u sitecheck go mod tidy

# Сборка приложения (собираем весь проект, не только main.go)
sudo -u sitecheck go build -o sitecheck .

# Настройка прав доступа
sudo chown sitecheck:sitecheck /opt/sitecheck/sitecheck
sudo chmod +x /opt/sitecheck/sitecheck

# Проверка сборки
ls -la /opt/sitecheck/sitecheck
file /opt/sitecheck/sitecheck

# Тестовый запуск (должен показать ошибку о DATABASE_URL - это нормально)
sudo -u sitecheck /opt/sitecheck/sitecheck || echo "Ошибка DATABASE_URL - ожидаемо"
```

### 8. Создание systemd сервиса

```bash
# Создание файла сервиса
sudo tee /etc/systemd/system/sitecheck.service << 'EOF'
[Unit]
Description=Site Check Service
After=network.target mysql.service

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
Environment=DATABASE_URL=sitecheck_user:SecurePassword123!@tcp(localhost:3306)/sitecheck
Environment=USE_AI=false
# Environment=OPENAI_API_KEY=sk-your-openai-api-key-here

# Настройки безопасности
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/sitecheck

[Install]
WantedBy=multi-user.target
EOF

# Перезагрузка конфигурации systemd
sudo systemctl daemon-reload

# Включение автозапуска сервиса
sudo systemctl enable sitecheck

# Проверка конфигурации сервиса
sudo systemctl cat sitecheck
```

### 9. Конфигурация веб-сервера

```bash
# Удаление конфигурации по умолчанию
sudo rm -f /etc/nginx/sites-enabled/default

# Создание конфигурации для Site Check
sudo tee /etc/nginx/sites-available/sitecheck.domain.com << 'EOF'
server {
    listen 80;
    server_name sitecheck.domain.com;

    # Логи
    access_log /var/log/nginx/sitecheck.domain.com.access.log;
    error_log /var/log/nginx/sitecheck.domain.com.error.log;

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

# Активация сайта
sudo ln -s /etc/nginx/sites-available/sitecheck.domain.com /etc/nginx/sites-enabled/

# Проверка конфигурации Nginx
sudo nginx -t

# Перезапуск Nginx
sudo systemctl restart nginx
sudo systemctl status nginx
```

### 10. Настройка DNS

```bash
# Проверка текущего IP сервера
curl -4 ifconfig.me
echo ""

# Проверка DNS записи (выполните на локальной машине или другом сервере)
# nslookup sitecheck.domain.com
# dig sitecheck.domain.com

echo "Настройте DNS запись:"
echo "Тип: A"
echo "Имя: sitecheck.domain.com"
echo "Значение: $(curl -4 -s ifconfig.me)"
echo "TTL: 300"
echo ""
echo "После настройки DNS проверьте распространение:"

# Ожидание распространения DNS
while true; do
    if nslookup sitecheck.domain.com | grep -q "$(curl -4 -s ifconfig.me)"; then
        echo "DNS запись распространилась успешно!"
        break
    else
        echo "Ожидание распространения DNS... (проверка каждые 30 сек)"
        sleep 30
    fi
done
```

### 11. Получение SSL сертификата

```bash
# Получение SSL сертификата от Let's Encrypt
sudo certbot --nginx -d sitecheck.domain.com \
  --non-interactive \
  --agree-tos \
  --email admin@domain.com

# Проверка статуса сертификата
sudo certbot certificates

# Настройка автоматического обновления
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
sudo systemctl status certbot.timer

# Тестирование автообновления
sudo certbot renew --dry-run

# Проверка конфигурации Nginx после SSL
sudo nginx -t
sudo systemctl reload nginx

# Проверка SSL сертификата
curl -I https://sitecheck.domain.com/healthz
```

### 12. Настройка файрвола

```bash
# Сброс настроек UFW (осторожно!)
sudo ufw --force reset

# Настройка политик по умолчанию
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Разрешение SSH (ВАЖНО: сделайте это до активации UFW!)
sudo ufw allow ssh
sudo ufw allow 22/tcp

# Разрешение HTTP и HTTPS
sudo ufw allow 'Nginx Full'
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Опционально: разрешение только для определенных IP
# sudo ufw allow from 192.168.1.0/24 to any port 22

# Активация файрвола
sudo ufw --force enable

# Проверка статуса
sudo ufw status verbose
sudo ufw status numbered

# Проверка логов файрвола
sudo tail -f /var/log/ufw.log
```

### 13. Запуск и проверка сервисов

```bash
# Перезагрузка конфигурации systemd
sudo systemctl daemon-reload

# Запуск MySQL (если не запущен)
sudo systemctl start mysql
sudo systemctl enable mysql

# Запуск сервиса Site Check
sudo systemctl start sitecheck
sudo systemctl enable sitecheck

# Запуск Nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Проверка статуса всех сервисов
echo "=== Статус MySQL ==="
sudo systemctl status mysql --no-pager

echo "=== Статус Site Check ==="
sudo systemctl status sitecheck --no-pager

echo "=== Статус Nginx ==="
sudo systemctl status nginx --no-pager

# Проверка портов
echo "=== Открытые порты ==="
sudo netstat -tlnp | grep -E ':80|:443|:3306|:8080'

# Проверка логов
echo "=== Последние логи Site Check ==="
sudo journalctl -u sitecheck --no-pager -n 10
```

### 14. Проверка работоспособности

```bash
# Проверка подключения к базе данных
mysql -u sitecheck_user -p sitecheck -e "SELECT COUNT(*) as api_logs_count FROM api_logs; SELECT COUNT(*) as ai_logs_count FROM ai_logs;"
# Введите пароль: SecurePassword123!

# Health check через локальный порт
curl -v http://localhost:8080/healthz

# Health check через домен (HTTP)
curl -v http://sitecheck.domain.com/healthz

# Health check через домен (HTTPS)
curl -v https://sitecheck.domain.com/healthz

# Тестовый запрос к API
curl -X POST https://sitecheck.domain.com/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://google.com"}' \
  -v

# Проверка записей в базе данных после API запроса
mysql -u sitecheck_user -p sitecheck -e "
SELECT id, route, url, status_code, created_at 
FROM api_logs 
ORDER BY created_at DESC 
LIMIT 5;
"

# Проверка логов приложения
echo "=== Логи приложения ==="
sudo tail -n 20 /var/log/sitecheck/sitecheck.log

# Проверка логов Nginx
echo "=== Access логи Nginx ==="
sudo tail -n 10 /var/log/nginx/sitecheck.domain.com.access.log

echo "=== Error логи Nginx ==="
sudo tail -n 10 /var/log/nginx/sitecheck.domain.com.error.log

# Проверка использования ресурсов
echo "=== Использование ресурсов ==="
free -h
df -h
ps aux | grep -E 'sitecheck|mysql|nginx' | grep -v grep
```

## Настройка OpenAI API (опционально)

Для включения AI-анализа настройте переменные окружения:

### Способ 1: Override файл systemd

```bash
# Создание override файла для сервиса
sudo systemctl edit sitecheck

# В открывшемся редакторе добавьте:
```

```ini
[Service]
Environment=USE_AI=true
Environment=OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

```bash
# Сохраните файл (Ctrl+X, Y, Enter) и перезапустите сервис
sudo systemctl daemon-reload
sudo systemctl restart sitecheck
sudo systemctl status sitecheck
```

### Способ 2: Прямое редактирование файла сервиса

```bash
# Остановка сервиса
sudo systemctl stop sitecheck

# Редактирование файла сервиса
sudo nano /etc/systemd/system/sitecheck.service

# Найдите и измените строки:
# Environment=USE_AI=false  →  Environment=USE_AI=true
# # Environment=OPENAI_API_KEY=sk-your-openai-api-key-here  →  Environment=OPENAI_API_KEY=sk-your-actual-key

# Перезагрузка и запуск
sudo systemctl daemon-reload
sudo systemctl start sitecheck
sudo systemctl status sitecheck
```

### Способ 3: Файл окружения

```bash
# Создание файла окружения
sudo tee /opt/sitecheck/.env << 'EOF'
DATABASE_URL=sitecheck_user:SecurePassword123!@tcp(localhost:3306)/sitecheck
USE_AI=true
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
EOF

# Настройка прав доступа
sudo chown sitecheck:sitecheck /opt/sitecheck/.env
sudo chmod 600 /opt/sitecheck/.env

# Создание override для использования .env файла
sudo systemctl edit sitecheck

# В редакторе добавьте:
```

```ini
[Service]
EnvironmentFile=/opt/sitecheck/.env
```

```bash
# Перезапуск сервиса
sudo systemctl daemon-reload
sudo systemctl restart sitecheck

# Проверка переменных окружения
sudo systemctl show sitecheck --property=Environment
```

## Обновление приложения

```bash
# Остановка сервиса
sudo systemctl stop sitecheck

# Переход в директорию проекта
cd /opt/sitecheck

# Создание резервной копии текущей версии
sudo -u sitecheck cp sitecheck sitecheck.backup.$(date +%Y%m%d_%H%M%S)

# Обновление исходного кода
sudo -u sitecheck git fetch origin
sudo -u sitecheck git pull origin master

# Проверка изменений
sudo -u sitecheck git log --oneline -5

# Обновление зависимостей
sudo -u sitecheck go mod tidy

# Пересборка приложения (собираем весь проект)
sudo -u sitecheck go build -o sitecheck .

# Настройка прав доступа
sudo chown sitecheck:sitecheck /opt/sitecheck/sitecheck
sudo chmod +x /opt/sitecheck/sitecheck

# Запуск сервиса
sudo systemctl start sitecheck

# Проверка статуса
sudo systemctl status sitecheck

# Проверка работоспособности
curl -v https://sitecheck.domain.com/healthz

# Проверка логов
sudo journalctl -u sitecheck --no-pager -n 10

echo "Обновление завершено!"
```

## Мониторинг и логи

### Логи приложения
- Основные логи: в файле в директории логов
- Системные логи: через journalctl

### Логи веб-сервера
- Access логи: запросы к API
- Error логи: ошибки проксирования

### Логи базы данных
- Логи MySQL: `/var/log/mysql/error.log`
- Медленные запросы: настройте `slow_query_log`
- Мониторинг производительности через `SHOW PROCESSLIST`

### Мониторинг состояния
- Статус сервисов через systemctl (приложение, MySQL, Nginx)
- Health check endpoint для автоматического мониторинга
- Мониторинг использования ресурсов
- Мониторинг размера базы данных и таблиц логов
- Проверка подключений к базе данных

## Безопасность

### Рекомендации по безопасности:
1. Регулярно обновляйте систему и пакеты
2. Используйте сильные пароли и SSH-ключи
3. Мониторьте логи на подозрительную активность
4. Настройте автоматические обновления безопасности
5. Ограничьте доступ к серверу только необходимыми портами
6. Регулярно создавайте резервные копии конфигурации

### Безопасность базы данных:
1. Используйте сильные пароли для пользователей MySQL
2. Ограничьте сетевой доступ к MySQL (bind-address = 127.0.0.1)
3. Отключите удаленный root доступ
4. Регулярно создавайте резервные копии базы данных
5. Настройте ротацию логов для предотвращения переполнения диска
6. Мониторьте размер таблиц логов и настройте их очистку

### Дополнительные меры:
- Настройка fail2ban для защиты от брутфорса
- Использование нестандартного SSH порта
- Настройка системы обнаружения вторжений
- Регулярный аудит безопасности

## Производительность

### Оптимизация производительности:
- Настройка лимитов ресурсов для сервиса
- Оптимизация конфигурации Nginx
- Мониторинг использования памяти и CPU
- Настройка ротации логов

### Оптимизация базы данных:
- Настройка индексов для часто используемых запросов
- Регулярная очистка старых записей из таблиц логов
- Оптимизация настроек MySQL (innodb_buffer_pool_size, query_cache)
- Мониторинг медленных запросов
- Архивирование старых данных

### Масштабирование:
- Запуск нескольких экземпляров сервиса
- Настройка load balancer
- Кэширование результатов API запросов
- Использование CDN для статических ресурсов
- Репликация базы данных для чтения
- Шардинг таблиц логов по дате
