# Устранение проблем с systemd сервисом

## Проблема: "Found modifications outside of the staging area"

Эта ошибка возникает при использовании `systemctl edit` когда файл остается пустым после редактирования.

### Быстрое решение

```bash
# Способ 1: Создание override файла вручную
mkdir -p /etc/systemd/system/sitecheck.service.d
cat > /etc/systemd/system/sitecheck.service.d/override.conf << 'EOF'
[Service]
Environment=USE_AI=true
Environment=OPENAI_API_KEY=sk-your-actual-openai-api-key-here
EOF

systemctl daemon-reload
systemctl restart sitecheck
```

### Альтернативные способы

#### Способ 2: Прямое редактирование основного файла
```bash
systemctl stop sitecheck
nano /etc/systemd/system/sitecheck.service

# Найдите и измените строки:
# Environment=USE_AI=false  →  Environment=USE_AI=true
# # Environment=OPENAI_API_KEY=...  →  Environment=OPENAI_API_KEY=sk-your-key

systemctl daemon-reload
systemctl start sitecheck
```

#### Способ 3: Использование файла окружения
```bash
# Создаем .env файл
cat > /opt/sitecheck/.env << 'EOF'
USE_AI=true
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
EOF

# Создаем override для использования .env файла
mkdir -p /etc/systemd/system/sitecheck.service.d
cat > /etc/systemd/system/sitecheck.service.d/override.conf << 'EOF'
[Service]
EnvironmentFile=/opt/sitecheck/.env
EOF

systemctl daemon-reload
systemctl restart sitecheck
```

## Проверка конфигурации

### Просмотр текущих настроек
```bash
# Показать полную конфигурацию сервиса
systemctl cat sitecheck

# Показать только override файлы
systemctl show sitecheck --property=Environment

# Проверить статус сервиса
systemctl status sitecheck
```

### Просмотр логов
```bash
# Логи сервиса в реальном времени
journalctl -u sitecheck -f

# Последние 50 строк логов
journalctl -u sitecheck -n 50

# Логи приложения
tail -f /var/log/sitecheck/sitecheck.log
```

## Проверка работы OpenAI API

После настройки переменных окружения проверьте, что AI включен:

```bash
# Проверьте логи на наличие сообщения об OpenAI API
journalctl -u sitecheck | grep -i openai

# Тестовый запрос к API
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{"url": "https://google.com"}' | jq

# В ответе поле "source" должно быть "ai" вместо "heuristic"
```

## Частые ошибки

### 1. Пустой override файл
```bash
# Проверить содержимое
cat /etc/systemd/system/sitecheck.service.d/override.conf

# Если файл пустой - удалить и создать заново
rm -f /etc/systemd/system/sitecheck.service.d/override.conf
# Затем создать заново одним из способов выше
```

### 2. Неправильный формат API ключа
```bash
# OpenAI API ключи начинаются с "sk-"
# Пример правильного ключа: sk-proj-1234567890abcdef...

# Проверить переменные окружения процесса
ps aux | grep sitecheck
# Найти PID процесса, затем:
cat /proc/PID/environ | tr '\0' '\n' | grep -E "(USE_AI|OPENAI)"
```

### 3. Права доступа к файлам
```bash
# Проверить права на .env файл
ls -la /opt/sitecheck/.env
chown sitecheck:sitecheck /opt/sitecheck/.env
chmod 600 /opt/sitecheck/.env

# Проверить права на override файл
ls -la /etc/systemd/system/sitecheck.service.d/
chmod 644 /etc/systemd/system/sitecheck.service.d/override.conf
```

## Полезные команды

```bash
# Перезагрузка конфигурации systemd
systemctl daemon-reload

# Перезапуск сервиса
systemctl restart sitecheck

# Проверка синтаксиса конфигурации
systemd-analyze verify sitecheck.service

# Показать зависимости сервиса
systemctl list-dependencies sitecheck
```
