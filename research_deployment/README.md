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
| `code/load_test/`, `outputs/load_test/` | Load-test client and, for each of the two matched free-tier services (FP32 and INT8-dynamic), every request (`requests.csv`), per-phase statistics (`summary.json`) and run settings (`run_info.json`) | Sections 3.2, 4.1; Tables 3–4; Figures 3–4 |
| `code/load_test/e1b_open_loop.py`, `e1b_analyse.py`, `outputs/load_test/open_loop/` | Open-loop test of 21 Nov 2026: Poisson arrivals at 8, 12 and 14 per minute (and 18 on INT8-dynamic) sent to both services at once, with the same per-request logs as above; `e1b_analysis.json` compares each rate's mean response with a replay through one first-come-first-served server and with the M/D/1 queue (`python -I e1b_analyse.py fp32 int8_dynamic` from that folder) | Sections 3.2, 4.1, 4.6; Table S12 |
| `code/analysis/compression_gradcam/`, `outputs/compression_gradcam/` | Kaggle notebook atik68/neuroscan-p2-compression-gradcam: FP16 and INT8 copies, agreement and accuracy differences with patient-cluster bootstrap intervals, latency, exactness and cost of the closed-form Grad-CAM, maps under compression. INT8-dynamic results in this folder come from 16-slice batches (`*_batched`) | Sections 3.3–3.4, 4.2–4.3; Tables 5, 7; Table S10 |
| `code/analysis/int8dyn_served/`, `outputs/int8dyn_served/` | Kaggle notebook atik68/neuroscan-p2-int8dyn-served: INT8-dynamic re-run one slice per request, as the server runs it, on all 10,602 held-out slices, with the same comparisons. These are the INT8-dynamic results reported in the paper | Sections 4.2–4.3; Tables 5–7 |
| `code/analysis/static_calibration/`, `outputs/static_calibration/` | Kaggle notebook atik68/neuroscan-p2-static-calibration: three further static INT8 copies (percentile calibration, entropy calibration, and min–max with the stem and downsampling convolutions in FP32), with a min–max rebuild that reproduces the paper's INT8-static file byte for byte; agreement, changed decisions, maps and model-call time on all 10,602 held-out slices (`per_row_probs.csv` holds every slice's probabilities) | Sections 3.3, 4.2–4.3; Tables 5–7 |
| `code/analysis/batch_sensitivity/`, `outputs/batch_sensitivity/` | Kaggle notebook atik68/neuroscan-p2-batch-sensitivity: how far INT8-dynamic's outputs depend on the batch, on all 10,602 held-out slices: the served four views run alone, batches of 2 to 32 slices, and 476 target slices each batched with 12 partners. `per_row_probs.csv` holds every slice's probabilities under each condition, and `map_differences.json`, from `map_figures.py`, the map differences with and without the 173 near-flat ISLES 2022 slices. The full maps stay in the notebook's output | Sections 4.2–4.3, 4.6; Table S2 |
| `code/analysis/fidelity_detail/`, `outputs/fidelity_detail_batched/` | Kaggle notebook atik68/neuroscan-p2-fidelity-detail: the audit of changed decisions (every changed slice with its true label, dataset and FP32 margin; per-dataset sensitivity and specificity; patient-level ischemic detection), first run on the batched outputs. The served re-run is in `outputs/int8dyn_served/` | Section 4.2; Table 6 |
| `code/analysis/peak_memory/`, `outputs/peak_memory/` | Kaggle notebook atik68/neuroscan-p2-peak-memory: whole-service peak memory by the kernel's high-water mark, and the control run on the `ru_maxrss` counter | Sections 4.1–4.2; Table 5 |
| `code/analysis/fig4_cases/`, `outputs/fig4_cases/` | Kaggle notebook atik68/neuroscan-p2-fig4-cases: the five Figure 5 cases, chosen by fixed rules (the input slices themselves are not included) | Figure 5 |
| `code/analysis/cluster_sensitivity/`, `outputs/cluster_sensitivity/` | Kaggle notebook atik68/neuroscan-p2-cluster-sensitivity (revision): agreement and accuracy intervals with Kaggle_CT excluded, grouped into file-number blocks, or one cluster per class folder, from the saved per-slice outputs | Section 4.2; Table S1 |
| `code/analysis/fp16_rootcause/`, `outputs/fp16_rootcause/` | Kaggle notebook atik68/neuroscan-p2-fp16-rootcause (revision): why the two FP16 copies do not load (the failing Grad-CAM nodes), opset and runtime checks, and repaired FP16 copies with their agreement, size and speed | Section 4.2; Table S3 |
| `code/analysis/memory_breakdown/`, `outputs/memory_breakdown/` | Kaggle notebook atik68/neuroscan-p2-memory-breakdown (revision): memory by stage in fresh processes (interpreter, libraries, server, loaded model, request), with the memory arena off and on | Section 4.1; Table S4 |
| `code/analysis/gradcam_arch/`, `outputs/gradcam_arch/` | Kaggle notebook atik68/neuroscan-p2-gradcam-arch (revision): closed-form Grad-CAM against autograd and its cost on five further ImageNet architectures, and the batch dependence of their INT8-dynamic copies; `v2/` is the re-run that also quantized Swin-T | Sections 4.2–4.3; Table S5 |
| `code/analysis/mixed_precision/`, `outputs/mixed_precision/` | Kaggle notebook atik68/neuroscan-p2-mixed-precision (revision): static INT8 copies with the layers most affected by rounding kept in FP32 (`layer_sensitivity.csv`), scored on all 10,602 held-out slices (`per_row_probs.csv`) | Sections 3.3, 4.2; Table S6 |
| `code/analysis/single_slice/`, `outputs/single_slice/` | Kaggle notebook atik68/neuroscan-p2-single-slice-int8 (revision): every held-out CT slice sent without its neighbours, FP32 and INT8-dynamic one request each, with the changed slices and per-slice probabilities | Section 4.2; Tables S7, S7b |
| `code/security/`, `outputs/security/` | Scripted test suite, sqlmap and ZAP drivers, run only on a local copy of the platform with invented accounts; results of the first run on the original code (`1_…`), the full suite on the original code (`2_…`), and the fixed code (`3_…`, including the administrator ZAP scan after the second round of fixes); `4_released_code/` is the suite on a fresh local rebuild of the released web application in `../webapp/`, with accounts made by `seed_test_accounts.py` | Sections 3.5, 4.4; Table S8 |
| `code/rebuild/`, `outputs/rebuild/` | Rebuild check: clone, fingerprints, environment, model files and answers on the 28 test-pack cases. `16ad849_first_rebuild/` exposed the unpinned dependencies; `1d5cce0_lock_file/` is the rebuild with the lock file reported in the paper | Sections 3.6, 4.5; Table S11 |
| `code/figures/` | Scripts that draw the figures: `make_fig1.py` and `make_fig2.py` (Figures 1–2), `make_fig3_4.py` (Figures 3–4), `make_fig4.py` (Figure 5); `make_fig3.py` drew the earlier combined load-test figure | Figures 1–5 |
| `SHA256SUMS.txt` | SHA-256 fingerprint of every file in this folder | — |

## Notes

- Where a folder has a `*_src.py` file, its Kaggle notebook is built from it by `build_kernel.py`, which adds the
  repository's served inference code; the other notebook scripts are pushed as they are. `kernel-metadata.json` names
  each notebook and its inputs.
- The API key was read from the environment (load test) or generated at random for each run (rebuild, memory check)
  and never written to any file. The security tests read their invented accounts from a local file that is not
  included.
- In `outputs/load_test/*/run_info.json` the services' host names are replaced by placeholders; nothing else in the
  outputs was changed.
