# Быстрый старт Site Check

## 🚀 Автоматическая установка (рекомендуется)

1. **Подключитесь к серверу:**
   ```bash
   ssh root@5.129.234.157
   ```

2. **Запустите автоматическую установку:**
   ```bash
   curl -sSL https://raw.githubusercontent.com/gocpa/ads-site-check/master/deploy.sh | bash
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
