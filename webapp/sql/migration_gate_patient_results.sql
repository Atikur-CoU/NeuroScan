-- Adds the clinician approval gate setting, switched off. Run once on an existing database.
-- Switch it on under Admin > Inference endpoint settings.
INSERT INTO settings (setting_key, setting_value) VALUES ('gate_patient_results', '0')
  ON DUPLICATE KEY UPDATE setting_key = setting_key;
