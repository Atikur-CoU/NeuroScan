-- NeuroScan clinical web app schema.
-- MySQL 5.7+/8.0, InnoDB, utf8mb4. All timestamps are DATETIME written in UTC by the app.

SET NAMES utf8mb4;

CREATE TABLE users (
  id                 INT UNSIGNED NOT NULL AUTO_INCREMENT,
  role               ENUM('patient','radiologist','doctor','admin') NOT NULL,
  full_name          VARCHAR(150) NOT NULL,
  email              VARCHAR(190) NOT NULL,
  password_hash      VARCHAR(255) NOT NULL,
  phone              VARCHAR(30) NULL,
  license_number     VARCHAR(100) NULL,
  is_active          TINYINT(1) NOT NULL DEFAULT 1,
  must_change_password TINYINT(1) NOT NULL DEFAULT 0,
  failed_login_count SMALLINT UNSIGNED NOT NULL DEFAULT 0,
  locked_until       DATETIME NULL,
  created_by         INT UNSIGNED NULL,
  created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_users_email (email),
  KEY idx_users_role (role),
  CONSTRAINT fk_users_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE patient_profiles (
  user_id       INT UNSIGNED NOT NULL,
  date_of_birth DATE NULL,
  sex           ENUM('male','female','other','unspecified') NULL DEFAULT 'unspecified',
  mrn           VARCHAR(50) NULL,
  blood_group   VARCHAR(5) NULL,
  notes         TEXT NULL,
  PRIMARY KEY (user_id),
  UNIQUE KEY uq_patient_profiles_mrn (mrn),
  CONSTRAINT fk_patient_profiles_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE doctor_patient (
  id                INT UNSIGNED NOT NULL AUTO_INCREMENT,
  patient_id        INT UNSIGNED NOT NULL,
  provider_id       INT UNSIGNED NOT NULL,
  relationship_type ENUM('primary_doctor','radiologist','consulting_doctor') NOT NULL DEFAULT 'primary_doctor',
  assigned_by       INT UNSIGNED NOT NULL,
  assigned_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  is_active         TINYINT(1) NOT NULL DEFAULT 1,
  PRIMARY KEY (id),
  UNIQUE KEY uq_doctor_patient (patient_id, provider_id, relationship_type),
  KEY idx_doctor_patient_provider (provider_id, is_active),
  KEY idx_doctor_patient_patient (patient_id, is_active),
  CONSTRAINT fk_dp_patient FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT fk_dp_provider FOREIGN KEY (provider_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT fk_dp_assigned_by FOREIGN KEY (assigned_by) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE scans (
  id                INT UNSIGNED NOT NULL AUTO_INCREMENT,
  public_uuid       CHAR(36) NOT NULL,
  patient_id        INT UNSIGNED NOT NULL,
  uploaded_by       INT UNSIGNED NOT NULL,
  modality          ENUM('CT','MRI') NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  stored_filename   VARCHAR(255) NOT NULL,
  storage_path      VARCHAR(500) NOT NULL,
  file_size_bytes   INT UNSIGNED NOT NULL,
  mime_type         VARCHAR(100) NOT NULL,
  sha256_hash       CHAR(64) NOT NULL,
  status            ENUM('pending','processing','completed','failed') NOT NULL DEFAULT 'pending',
  notes             VARCHAR(500) NULL,
  uploaded_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_scans_public_uuid (public_uuid),
  KEY idx_scans_patient (patient_id, uploaded_at),
  KEY idx_scans_status (status),
  CONSTRAINT fk_scans_patient FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT fk_scans_uploaded_by FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE predictions (
  id                     INT UNSIGNED NOT NULL AUTO_INCREMENT,
  scan_id                INT UNSIGNED NOT NULL,
  prediction             ENUM('normal','ischemic','hemorrhagic') NULL,
  prob_normal            DECIMAL(6,5) NULL,
  prob_ischemic          DECIMAL(6,5) NULL,
  prob_hemorrhagic       DECIMAL(6,5) NULL,
  model_version          VARCHAR(100) NULL,
  inference_endpoint     VARCHAR(255) NULL,
  attempt_count          SMALLINT UNSIGNED NOT NULL DEFAULT 0,
  last_error             VARCHAR(500) NULL,
  request_started_at     DATETIME NULL,
  request_completed_at   DATETIME NULL,
  created_at             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_predictions_scan (scan_id),
  CONSTRAINT fk_predictions_scan FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE reviews (
  id                 INT UNSIGNED NOT NULL AUTO_INCREMENT,
  scan_id            INT UNSIGNED NOT NULL,
  reviewer_id        INT UNSIGNED NOT NULL,
  reviewer_role      ENUM('doctor','radiologist') NOT NULL,
  agrees_with_ai     TINYINT(1) NULL,
  diagnosis_override ENUM('normal','ischemic','hemorrhagic','other','inconclusive') NULL,
  severity           ENUM('low','moderate','high','critical') NULL,
  notes              TEXT NULL,
  status             ENUM('draft','final') NOT NULL DEFAULT 'draft',
  amends_review_id   INT UNSIGNED NULL,
  created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_reviews_scan (scan_id),
  KEY idx_reviews_reviewer (reviewer_id),
  CONSTRAINT fk_reviews_scan FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
  CONSTRAINT fk_reviews_reviewer FOREIGN KEY (reviewer_id) REFERENCES users(id) ON DELETE CASCADE,
  CONSTRAINT fk_reviews_amends FOREIGN KEY (amends_review_id) REFERENCES reviews(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE settings (
  id            INT UNSIGNED NOT NULL AUTO_INCREMENT,
  setting_key   VARCHAR(100) NOT NULL,
  setting_value TEXT NULL,
  updated_by    INT UNSIGNED NULL,
  updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_settings_key (setting_key),
  CONSTRAINT fk_settings_updated_by FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE audit_log (
  id             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  actor_user_id  INT UNSIGNED NULL,
  actor_role     VARCHAR(20) NULL,
  action         VARCHAR(100) NOT NULL,
  target_type    VARCHAR(50) NULL,
  target_id      VARCHAR(64) NULL,
  ip_address     VARCHAR(45) NULL,
  user_agent     VARCHAR(255) NULL,
  details        TEXT NULL,
  created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_audit_actor (actor_user_id),
  KEY idx_audit_action (action),
  KEY idx_audit_created (created_at),
  KEY idx_audit_target (target_type, target_id),
  CONSTRAINT fk_audit_actor FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE login_attempts (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  email        VARCHAR(190) NOT NULL,
  ip_address   VARCHAR(45) NOT NULL,
  success      TINYINT(1) NOT NULL,
  attempted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_login_attempts_email (email, attempted_at),
  KEY idx_login_attempts_ip (ip_address, attempted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE api_tokens (
  id            INT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id       INT UNSIGNED NOT NULL,
  token_hash    CHAR(64) NOT NULL,
  device_label  VARCHAR(100) NULL,
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_used_at  DATETIME NULL,
  expires_at    DATETIME NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_api_tokens_hash (token_hash),
  KEY idx_api_tokens_user (user_id),
  CONSTRAINT fk_api_tokens_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO settings (setting_key, setting_value) VALUES
  ('inference_endpoint_url', ''),
  ('inference_api_key', ''),
  ('inference_timeout_seconds', '30'),
  ('inference_connect_timeout_seconds', '8'),
  ('max_upload_mb', '10'),
  ('gate_patient_results', '0');
