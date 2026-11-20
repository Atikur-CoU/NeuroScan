-- v2 inference server (final model R30A s123, ConvNeXt-Tiny): optional context images and the input pathway.
-- scans.context_files: JSON object of extra stored files for this scan, e.g. {"prev": "...png", "next": "...png"}
--   for CT adjacent slices or {"adc": "...png"} for MRI; NULL when only one image was uploaded.
-- predictions.pathway: which input pathway the server ran (ct_adjacent_slices / mri_dwi_adc / single_image);
-- predictions.evaluated_pathway: 1 when that pathway is the one evaluated in the paper, 0 for the single-image fallback;
-- predictions.warnings: JSON array of scope warnings returned by the server; predictions.decision_note: set when the
--   predicted class is not the one with the largest displayed probability (locked decision weights).
-- All NULL for scans predicted before this migration; callers must treat them as optional.
ALTER TABLE scans ADD COLUMN context_files TEXT NULL AFTER storage_path;
ALTER TABLE predictions
  ADD COLUMN pathway VARCHAR(40) NULL AFTER likely_not_a_scan,
  ADD COLUMN evaluated_pathway TINYINT(1) NULL AFTER pathway,
  ADD COLUMN warnings TEXT NULL AFTER evaluated_pathway,
  ADD COLUMN decision_note VARCHAR(300) NULL AFTER warnings;
