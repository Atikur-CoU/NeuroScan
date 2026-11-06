"""Map differences of each batching against the served run, with and without the 173 ISLES 2022 normal slices whose
served probabilities and maps are identical. Run after the notebook, on its downloaded output:

    python -I map_figures.py <batch-sensitivity output folder> <INT8-dyn_served.npz> <map_differences.json>

The difference is the one the notebook uses: the largest absolute change over all cells of all three classes, relative
to the largest absolute value of the served map. File-order batches are left out, because their maps are not in the
notebook's output."""
import csv
import json
import sys

import numpy as np

CONDITIONS = ["views_alone", "random_b2_seed123", "random_b4_seed123", "random_b8_seed123", "random_b16_seed123",
              "random_b16_seed124", "random_b32_seed123"]
FLAT_PROBS = [0.720409, 0.196274, 0.083317]

out_dir, served_npz, out_json = sys.argv[1:4]
served = np.load(served_npz, allow_pickle=False)
cams, probs = served["cams"], served["probs"]
with open(f"{out_dir}/per_row_probs.csv", newline="") as f:
    rows = list(csv.DictReader(f))
n = len(rows)
csv_probs = np.array([[float(r[f"p_{k}_served_four_views_together"]) for k in ("normal", "ischemic", "hemorrhagic")] for r in rows])
assert np.abs(csv_probs - probs).max() < 1e-5, "row order differs between the two files"
flat = np.all(np.isclose(csv_probs, FLAT_PROBS, atol=1e-6), axis=1)
assert flat.sum() == 173
assert {r["dataset"] for r, f in zip(rows, flat) if f} == {"ISLES2022_dwi_adc"}
assert {r["label"] for r, f in zip(rows, flat) if f} == {"normal"}
peak = np.abs(cams).reshape(n, -1).max(1)


def summary(x):
    return {"median": float(np.median(x)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


res = {"flat_slices": int(flat.sum()),
       "distinct_served_maps_among_flat_slices": len(np.unique(np.round(cams[flat].reshape(flat.sum(), -1), 6), axis=0)),
       "served_map_peak": {"flat_median": float(np.median(peak[flat])), "others_median": float(np.median(peak[~flat]))},
       "conditions": {}}
for c in CONDITIONS:
    other = np.load(f"{out_dir}/partial/{c}.npz", allow_pickle=False)["cams"]
    rel = np.abs(other - cams).reshape(n, -1).max(1) / (peak + 1e-12)
    worst = int(np.where(~flat)[0][rel[~flat].argmax()])
    res["conditions"][c] = {"all_slices": summary(rel), "without_flat_slices": summary(rel[~flat]),
                            "above_0.3": int((rel > .3).sum()), "above_0.3_flat": int((rel[flat] > .3).sum()),
                            "largest_without_flat": {"dataset": rows[worst]["dataset"], "served_map_peak": float(peak[worst]),
                                                     "share_of_slices_with_a_lower_peak": float((peak < peak[worst]).mean())}}
with open(out_json, "w", newline="\n") as f:
    json.dump(res, f, indent=1)
for k, v in res["conditions"].items():
    print(k, round(v["all_slices"]["p99"], 3), round(v["without_flat_slices"]["p99"], 3), round(v["all_slices"]["max"], 3),
          round(v["without_flat_slices"]["max"], 3), v["largest_without_flat"]["dataset"], round(v["largest_without_flat"]["share_of_slices_with_a_lower_peak"], 3))
