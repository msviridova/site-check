CREATE DATABASE sitecheck CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'sitecheck'@'localhost' IDENTIFIED BY '1234509876';
GRANT ALL PRIVILEGES ON sitecheck.* TO 'sitecheck'@'localhost';
FLUSH PRIVILEGES;

-- сначала api_logs
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

-- потом ai_logs
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