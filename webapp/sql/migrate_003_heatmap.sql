-- Adds Grad-CAM heatmap storage to predictions. NULL for any scan
-- predicted before this migration (or by a heatmap-less inference
-- endpoint) -- callers must treat heatmap_path as optional.
ALTER TABLE predictions ADD COLUMN heatmap_path VARCHAR(500) NULL AFTER prob_hemorrhagic;
