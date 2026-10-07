# NeuroScan inference server (Render)

Always-on replacement for the Kaggle/Colab + ngrok notebook setup
(`kaggle_kernels/phase_o_inference_server/`).

**v2 (2026-10):** serves the paper's final model (R30A s123, ConvNeXt-Tiny)
as ONNX with ONNX Runtime — no PyTorch at run time (peak ~300 MB, ~0.85 s
per request on the free tier's one CPU). `/api/predict` accepts
`modality` (CT/MRI) plus optional `prev_file` + `next_file` (CT adjacent
slices) or `adc_file` (MRI); with them the evaluated input pathway runs,
with one image the single-image fallback runs and the response says so.
Older clients that send only `file` keep working. Decision rule: 4-view
TTA, plain argmax (the rule the paper evaluates the alert tiers under).
See the docstring of `inference.py`. `webapp/inference.py` is the old
ResNet18 version and is no longer kept in sync.

**Trade-off of the free tier**: the service spins down after 15 minutes
of no traffic and takes ~30-60s to wake on the next request. There is no
manual-restart routine like the notebook needed, though — once deployed,
it just works, cold start aside.

## Where the checkpoint comes from

The model checkpoint is downloaded at **container startup** (`start.sh`)
from a **GitHub Release asset on this same repo**, not baked into the
image at build time and not fetched from Kaggle.

Two earlier approaches were tried and abandoned, kept here as a record
so nobody re-discovers the same dead ends:

- **Baking it into the image via a Kaggle download at build time**
  failed with a 403 that looked like a build-arg-passing problem but
  wasn't — Render was correctly passing `KAGGLE_USERNAME`/`KAGGLE_KEY`
  into the build.
- **Downloading from Kaggle at runtime** (with the credentials verified,
  byte-for-byte, to be reaching the container correctly) *still* got a
  403 — from Render's network specifically, while the identical request
  worked flawlessly from elsewhere. Most likely Kaggle restricting its
  newer API token type from cloud-hosting IP ranges; never fully
  confirmed since that's not independently verifiable from outside
  Render's network.

GitHub was chosen instead because Render's own ability to reach GitHub
reliably is already proven — that's how it pulls this repo to build
from in the first place — so a GitHub Release asset doesn't introduce
a new unknown the way a second attempt at Kaggle would have.

## One-time setup

### 1. Get this folder into a Git repo

Already done if you're reading this from `Atikur-CoU/NeuroScan` — skip
to step 2. Otherwise:

```bash
cd render_inference_server
git init && git add . && git commit -m "Initial NeuroScan inference server"
```

Create a **private** repo on GitHub, then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

### 2. Publish the checkpoint as a GitHub Release asset

On github.com, go to the repo → **Releases** (right sidebar) → **Create a new release**:

1. **Tag**: `checkpoint-v2-convnext` (must match exactly — `start.sh` looks for this tag).
   The old `checkpoint-v1` release (Phase O ResNet18 `best.pt`) can stay
   for rollback together with the previous commit.
2. Title: anything (e.g. "Final model R30A s123 (ONNX)").
3. Upload `neuroscan_r30a_s123.onnx` (111 MB; produced by the Kaggle kernel
   `neuroscan-q-onnx-export-check`, SHA-256 `4937ab38…1502eb`, which
   `inference.py` verifies at start-up) with exactly that filename, since
   `start.sh` matches on it.
4. **Publish release.** The repo stays private; this asset is only
   downloadable with a valid token (next step), not by a public URL.

### 3. Generate a GitHub Personal Access Token

Github → Settings → Developer settings → **Personal access tokens** →
**Fine-grained tokens** → **Generate new token**:

- **Repository access**: Only select repositories → this one repo.
- **Permissions**: Contents → **Read-only** (that's all this needs).
- Copy the generated token (starts with `github_pat_...`) — you won't
  see it again after leaving the page.

### 4. Create (or update) the Render Web Service

1. [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service** → select this repo.
2. Confirm **Runtime: Docker**, **Instance Type: Free**.
3. Under **Environment Variables**, set:
   - `NEUROSCAN_API_KEY` — a random value (e.g. `python -c "import secrets; print(secrets.token_urlsafe(24))"`) — this is what the PHP site's Settings page will need too.
   - `GITHUB_TOKEN` — the fine-grained token from step 3.
   - Remove `KAGGLE_USERNAME`/`KAGGLE_KEY` if they're still there from an earlier attempt — no longer used.
4. Create/redeploy.

### 5. Point the site at it

Render gives you a permanent URL like `https://neuroscan-1w5e.onrender.com` (this one **doesn't rotate**, unlike the notebook's ngrok URL). Set it once in `/admin_settings.php` on the live site (endpoint URL + the `NEUROSCAN_API_KEY` value), and you shouldn't need to touch it again.

## Verifying it works

```bash
curl https://<your-service>.onrender.com/api/health \
  -H "X-API-Key: <your-key>"
```

Expect `{"status":"ok","checkpoint":{"epoch":18,"val_acc":0.9675...}}` — if `epoch` is anything else, the wrong file got uploaded as the release asset; re-check step 2.

The first request after any idle period will be slow (cold start) — that's expected on the free tier, not a bug.
