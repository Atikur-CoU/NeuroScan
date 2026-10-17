# NeuroScan — research materials

Code, data partitions, saved model predictions and analysis outputs behind the NeuroScan paper (multi-source CT and
diffusion-MRI stroke slice classification). The inference service in the repository root serves the final model; this
folder documents how that model was built and evaluated.

**No images and no model weights are included.** All imaging data are public and must be obtained from their original
sources (Table 1 of the paper). File paths inside the manifests refer to those sources.

## Contents

| Folder | What it holds | Paper |
|---|---|---|
| `manifests/split_v3/` | Patient-level training, validation and test partitions (one row per slice: source, label, file path, patient key) | Section 3.2; Table 1 |
| `manifests/phase2/` | Second-phase validation patients, RSNA patient map and split summary | Section 3.4 |
| `manifests/added_sources/` | Row lists of the sources added by the preparation scripts: AISD and RSNA within-source normals, SOOP, extra RSNA training slices | Section 3.2; Table 1 |
| `development_record/` | Dated development record, including the adoption rules and the final evaluation plan | Sections 3.4–3.5; Supplementary Table S7 |
| `code/preprocessing/` | Slice extraction and conversion per source, CPAISD Hounsfield-unit re-rendering, within-source normals, SOOP preparation, AISD DICOM and CPAISD count audits | Sections 3.2–3.3 |
| `code/training/` | Training scripts: final model (ConvNeXt-Tiny, seeds 123 and 7), ResNet18 reference, EfficientNet-B0, five comparison backbones, single-modality models, second-phase experiments | Section 3.4; Tables 3, 7, 9 |
| `code/analysis/` | Final evaluation, post-hoc analyses, single-image and Grad-CAM experiments, alert tiers, input-file check, backbone comparison, ONNX export and memory checks, figure-case kernels | Sections 3.5–4.6 |
| `outputs/` | JSON and CSV outputs of every analysis above | Tables 4–12; Supplementary Tables S2–S10 |
| `predictions/` | Saved class probabilities (validation, internal test, SOOP cohort) for every model in the paper | All results |
| `code/figures/` | Scripts that draw the paper's figures from the outputs above | Figures 1–9, S1–S3 |
| `SHA256SUMS.txt` | SHA-256 fingerprint of every file in this folder | Supplementary Section S3 |

## Key files

| Item | File |
|---|---|
| Final model, held-out probabilities | `predictions/final_convnext_tiny_seed123/predictions_test.csv`, `predictions_soop_external.csv` |
| Final evaluation (one-time scoring) | `code/analysis/final_evaluation/evaluate.py` → `outputs/final_evaluation/final_evaluation.json` |
| Backbone comparison | `code/analysis/backbone_comparison/backbones.py` → `outputs/backbone_comparison/backbone_comparison.json` |
| Single-image pathway and Grad-CAM | `code/analysis/single_image_and_gradcam/c2c3.py`; CPAISD re-run in `gradcam_cpaisd_identity/` |
| Alert tiers | `code/analysis/alert_tiers/tiers.py` → `outputs/alert_tiers/deployed_tiers.json` |
| Input-file check | `code/analysis/input_file_check/input_check.py` → `outputs/input_file_check/input_check_eval.json` |

The four prediction fingerprints quoted in Supplementary Section S3 can be checked against `SHA256SUMS.txt`
(for example `8ac2d98c…6fcb` for the final model's internal-test probabilities).

## Running the code

The scripts ran as Kaggle notebooks (training of the final model and of the comparison models on a local
workstation with the same scripts); input paths point to `/kaggle/input/...`. To rerun them, download the public
sources, rebuild the inputs with `code/preprocessing/`, and adjust the paths at the top of each script. The figure
scripts read the files in `outputs/` and `predictions/` and use local paths that need the same adjustment.
Software versions are listed in Supplementary Section S2.

## Notes

- The training, validation and test rows of the final model are `manifests/split_v3/` plus `manifests/added_sources/`.
  In `soop_rows.csv`, the `external` rows are the SOOP hold-out pool: 66 patients form the SOOP cohort and the remaining
  patients were moved into training by the rule in the training scripts (85%, fixed seed), giving 31,307 SOOP training rows
  and 1,672 SOOP cohort rows.
- `development_record/development_record.md` is a copy of the dated record with terminology harmonized; its first
  paragraph lists exactly what changed. No entry, date, number or decision was changed.
- Comments and output labels in the analysis code were harmonized to the terms used in the paper ("post-hoc");
  no code logic was changed.
- Model weights are not stored here because of file-size limits; the served ONNX model is fetched by the inference
  service at start-up and its SHA-256 fingerprint is checked before serving.
