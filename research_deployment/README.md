# NeuroScan — research materials for the deployment paper

Code, logs and results behind the second NeuroScan paper, which evaluates the deployment of the final model rather than
the model itself: a load test on Render's free tier, compressed copies of the served model, Grad-CAM computed in closed
form inside the ONNX graph, security tests of the platform, and a rebuild of the service from this repository. The
model, its training and its evaluation are documented in `../research/`.

**No images and no model weights are included.** The held-out slices are public and come from the sources listed in
`../research/`. The FP32 model is the release asset of tag `checkpoint-v2-convnext`, and the INT8-dynamic copy the
release asset of tag `paper2-int8-dynamic`.

## Contents

| Folder | What it holds | Paper |
|---|---|---|
| `code/load_test/`, `outputs/load_test/` | Load-test client and, for each of the two matched free-tier services (FP32 and INT8-dynamic), every request (`requests.csv`), per-phase statistics (`summary.json`) and run settings (`run_info.json`) | Sections 4.1, 5.1; Tables 3–4; Figure 3 |
| `code/analysis/compression_gradcam/`, `outputs/compression_gradcam/` | Kaggle notebook atik68/neuroscan-p2-compression-gradcam: FP16 and INT8 copies, agreement and accuracy differences with patient-cluster bootstrap intervals, latency, exactness and cost of the closed-form Grad-CAM, maps under compression. INT8-dynamic results in this folder come from 16-slice batches (`*_batched`) | Sections 4.2–4.3, 5.2–5.3; Tables 5, 8, 9 |
| `code/analysis/int8dyn_served/`, `outputs/int8dyn_served/` | Kaggle notebook atik68/neuroscan-p2-int8dyn-served: INT8-dynamic re-run one slice per request, as the server runs it, on all 10,602 held-out slices, with the same comparisons. These are the INT8-dynamic results reported in the paper | Sections 5.2–5.3; Tables 5, 6, 9 |
| `code/analysis/static_calibration/`, `outputs/static_calibration/` | Kaggle notebook atik68/neuroscan-p2-static-calibration: three further static INT8 copies (percentile calibration, entropy calibration, and min–max with the stem and downsampling convolutions in FP32), with a min–max rebuild that reproduces the paper's INT8-static file byte for byte; agreement, changed decisions, maps and model-call time on all 10,602 held-out slices (`per_row_probs.csv` holds every slice's probabilities) | Sections 4.2, 5.2–5.3; Tables 5, 6, 9 |
| `code/analysis/batch_sensitivity/`, `outputs/batch_sensitivity/` | Kaggle notebook atik68/neuroscan-p2-batch-sensitivity: how far INT8-dynamic's outputs depend on the batch, on all 10,602 held-out slices: the served four views run alone, batches of 2 to 32 slices, and 476 target slices each batched with 12 partners. `per_row_probs.csv` holds every slice's probabilities under each condition, and `map_differences.json`, from `map_figures.py`, the map differences with and without the 173 near-flat ISLES 2022 slices. The full maps stay in the notebook's output | Sections 4.2, 5.2–5.3; Table 7 |
| `code/analysis/fidelity_detail/`, `outputs/fidelity_detail_batched/` | Kaggle notebook atik68/neuroscan-p2-fidelity-detail: the audit of changed decisions (every changed slice with its true label, dataset and FP32 margin; per-dataset sensitivity and specificity; patient-level ischemic detection), first run on the batched outputs. The served re-run is in `outputs/int8dyn_served/` | Section 5.2; Table 6 |
| `code/analysis/peak_memory/`, `outputs/peak_memory/` | Kaggle notebook atik68/neuroscan-p2-peak-memory: whole-service peak memory by the kernel's high-water mark, and the control run on the `ru_maxrss` counter | Section 5.2; Table 5 |
| `code/analysis/fig4_cases/`, `outputs/fig4_cases/` | Kaggle notebook atik68/neuroscan-p2-fig4-cases: the five Figure 4 cases, chosen by fixed rules (the input slices themselves are not included) | Figure 4 |
| `code/security/`, `outputs/security/` | Scripted test suite, sqlmap and ZAP drivers, run only on a local copy of the platform with invented accounts; results of the first run on the original code (`1_…`), the full suite on the original code (`2_…`), and the fixed code (`3_…`, including the administrator ZAP scan after the second round of fixes) | Sections 4.4, 5.4; Table 10 |
| `code/rebuild/`, `outputs/rebuild/` | Rebuild check: clone, fingerprints, environment, model files and answers on the 28 test-pack cases. `16ad849_first_rebuild/` exposed the unpinned dependencies; `1d5cce0_lock_file/` is the rebuild with the lock file reported in the paper | Sections 4.5, 5.5; Table 11 |
| `code/figures/` | Scripts that draw Figures 1–4 | Figures 1–4 |
| `SHA256SUMS.txt` | SHA-256 fingerprint of every file in this folder | — |

## Notes

- The Kaggle notebooks are built from the `*_src.py` files by `build_kernel.py`, which adds the repository's served
  inference code; `kernel-metadata.json` names each notebook and its inputs.
- The API key was read from the environment (load test) or generated at random for each run (rebuild, memory check)
  and never written to any file. The security tests read their invented accounts from a local file that is not
  included.
- In `outputs/load_test/*/run_info.json` the services' host names are replaced by placeholders; nothing else in the
  outputs was changed.
