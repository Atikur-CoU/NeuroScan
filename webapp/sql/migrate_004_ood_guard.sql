-- Adds an out-of-distribution guard flag to predictions: TRUE when the
-- uploaded image doesn't look like a grayscale CT/MRI scan (see
-- GRAD_CAM_HEATMAP.md and webapp/inference.py's _looks_like_grayscale_scan).
-- NULL for any scan predicted before this migration -- callers must treat
-- likely_not_a_scan as optional and NULL-safe (not simply falsy).
ALTER TABLE predictions ADD COLUMN likely_not_a_scan TINYINT(1) NULL AFTER heatmap_path;
