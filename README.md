# NeuroScan inference server (Render)

Always-on replacement for the Kaggle/Colab + ngrok notebook setup
(`kaggle_kernels/phase_o_inference_server/`). Same model, same
`inference.py` (byte-identical to `webapp/inference.py`), same
`/api/predict` + `/api/health` contract the PHP site and Android app
already expect — only the hosting mechanism changes.

**Trade-off of the free tier**: the service spins down after 15 minutes
of no traffic and takes ~30-60s to wake on the next request. There is no
manual-restart routine like the notebook needed, though — once deployed,
it just works, cold start aside.

The checkpoint is downloaded from Kaggle at **container startup**
(`start.sh`), not baked in at build time — this trades a few extra
seconds on each cold start for one less moving part to depend on.
(Note: an earlier version of this deployment failed with a `403
Forbidden - Permission 'datasets.get' was denied` error that looked
like a build-arg problem but wasn't — Render *was* correctly passing
`KAGGLE_USERNAME`/`KAGGLE_KEY` into the build. The real cause was the
pinned `kaggle==1.6.17` package predating Kaggle's newer `KGAT_`-prefixed
API token format; `requirements.txt` now pins `kaggle==2.2.4`, which
handles it correctly.)

## One-time setup

### 1. Get this folder into a Git repo

Render deploys from a Git repository (GitHub, GitLab, or Bitbucket), not
from a local folder — this piece is unavoidable and has to happen on
your end since it requires your own account:

```bash
cd render_inference_server
git init
git add .
git commit -m "Initial NeuroScan inference server"
```

Then create a new **private** repository on GitHub (github.com/new),
and push:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

### 2. Create the Render Web Service

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**.
2. Connect your GitHub account (if not already) and select the repo you just pushed.
3. Render should detect the `Dockerfile` automatically — confirm **Runtime: Docker**.
4. **Instance Type**: Free.
5. Under **Environment Variables**, add:
   - `NEUROSCAN_API_KEY` — generate one yourself (e.g. run `python -c "import secrets; print(secrets.token_urlsafe(24))"` locally) and paste the value. This is what the PHP site's Settings page will need too.
   - `KAGGLE_USERNAME` — your Kaggle username.
   - `KAGGLE_KEY` — your Kaggle API key (from `kaggle.json`, the `key` field — generate one at kaggle.com/settings if you don't have it handy).
6. Click **Create Web Service** and let it build (the build itself is quick — it only installs Python packages; the checkpoint downloads on first startup instead, so the very first request after deploy will take a bit longer than usual).

### 3. Point the site at it

Once the build succeeds, Render gives you a permanent URL like
`https://neuroscan-inference.onrender.com`. This one **doesn't rotate**
(unlike the ngrok URL) — set it once in `/admin_settings.php` on the
live site (endpoint URL + the `NEUROSCAN_API_KEY` value from step 2.5),
and you shouldn't need to touch it again.

## Verifying it works

```bash
curl https://<your-service>.onrender.com/api/health \
  -H "X-API-Key: <your-key>"
```

Expect `{"status":"ok","checkpoint":{"epoch":18,"val_acc":0.9675...}}` —
if `epoch` is anything else, the wrong checkpoint got downloaded; check
that the Kaggle dataset reference in `start.sh` (`kaggle datasets
download -d atik68/neuroscan-phase-o-checkpoints-final ...`) still
points at the right one.

The first request after any idle period will be slow (cold start) —
that's expected on the free tier, not a bug.
