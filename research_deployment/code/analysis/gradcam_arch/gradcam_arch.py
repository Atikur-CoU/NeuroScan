"""
NeuroScan paper 2 -- K6, JTCSST revision (R1-2, R1-6, R2-8): does the closed-form, in-graph Grad-CAM and the
batch dependence of INT8-dynamic carry over to other architectures?  CPU only; no stroke data, no NeuroScan weights.

Models: torchvision ImageNet weights (downloaded from download.pytorch.org), each with a 3-class head made of three
fixed rows of its ImageNet classifier (classes 1, 500, 999), so the head has the served model's form and size.
  pool_ln_linear  ConvNeXt-Tiny (control: the served architecture), ConvNeXt-Small, ConvNeXt-Base
                  (stage 4 -> global mean -> LayerNorm -> Linear, as in the paper)
  pool_linear     ResNet-50 (layer4), EfficientNet-B0 (features), Swin-T after its final LayerNorm
  not closed form Swin-T before its final LayerNorm: the LayerNorm acts at every position before pooling, so the
                  gradient differs between positions and no constant channel weight exists; reported for contrast.
Inputs: 32 deterministic head-like grayscale phantoms (the export-check kernel's generator), 224 x 224, identity view.

Per model:
  1. 3-class head logits vs the full torchvision model's logits for the same three classes.
  2. Closed-form Grad-CAM (PyTorch and exported ONNX, FP32) vs autograd Grad-CAM on the same layer, all three classes:
     max |difference| of max-normalised maps (the paper's measure, 4.2e-5 for the served model).
  3. Cost: ONNX Runtime 1.20.1, one thread (as served), median time of a logits-only graph vs logits + Grad-CAM.
  4. INT8-dynamic, the paper's recipe (quant_pre_process; quantize_dynamic, QInt8 weights, MatMul only, head nodes
     excluded): per-request vs batched (batches of 16) outputs on the 32 phantoms -- max |dp|, top-1 changes, map
     difference relative to the per-request map's peak; and the number of MatMul nodes quantized.
Every model is wrapped in try/except: a failure is recorded and the run continues. Output: JSON + CSV only (no
models are written to the output folder).

Version 2 (20 Nov 2026): reruns Swin-T after its final LayerNorm only. In version 1 its INT8 step stopped in
quant_pre_process ("Incomplete symbolic shape inference"); version 2 runs quant_pre_process with
skip_symbolic_shape=True (ONNX shape inference and graph optimisation are still applied) and, if that also fails,
quantizes the exported graph without pre-processing; the path taken is recorded. The working directory is /tmp, so
the pre-processing temporaries no longer land in the output. Outputs are named gradcam_arch_v2.*; version 1's
results for the other models stand.
"""
import os
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.chdir("/tmp")
T0 = time.time()
ONLY = {"swin_t_post_norm"}


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def pip(args):
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    log(f"pip {' '.join(args)}: exit {r.returncode} ({time.time() - t:.0f}s) {r.stderr[-300:]}")


pip(["onnxruntime==1.20.1", "onnx==1.18.0"])
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402
from torchvision import models  # noqa: E402

OUT = Path("/kaggle/working/output")
OUT.mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/k6")
WORK.mkdir(parents=True, exist_ok=True)
CLASSES = [1, 500, 999]
N_IMG, BATCH, N_TIME, MAX_IR = 32, 16, 30, 10
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
torch.set_num_threads(4)
R = {"note": "K6, revision; ImageNet weights with a 3-row head, synthetic phantoms; no stroke data",
     "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__, "onnx": onnx.__version__},
     "classes": CLASSES, "models": {}}
ROWS = []


def save():
    (OUT / "gradcam_arch_v2.json").write_text(json.dumps(R, indent=2, default=float))
    pd.DataFrame(ROWS).to_csv(OUT / "gradcam_arch_v2.csv", index=False)


def phantom(seed):
    """Same generator as kernel neuroscan-q-onnx-export-check: head-like ellipse, texture, one lesion-like blob."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", (512, 512), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([60, 40, 452, 472], fill=int(rng.integers(150, 200)))
    d.ellipse([80, 60, 432, 452], fill=int(rng.integers(90, 130)))
    cx, cy, r = rng.integers(160, 352), rng.integers(140, 372), rng.integers(20, 60)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.integers(40, 220)))
    a = np.asarray(img, dtype=np.float32) + rng.normal(0, 8, (512, 512))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))


def to_array(g):
    a = np.asarray(Image.merge("RGB", (g, g, g)), np.float32).transpose(2, 0, 1) / 255.0
    return (a - MEAN) / STD


X = np.stack([to_array(phantom(k).resize((224, 224))) for k in range(N_IMG)]).astype(np.float32)
XT = torch.from_numpy(X)
log(f"{N_IMG} phantoms built")


# ------------------------------------------------------------------ backbones and heads
class Split(nn.Module):
    """backbone(x) -> A (B, C, H, W): the Grad-CAM layer; head(A) -> 3 logits; cams(A, logits) -> closed form."""

    def __init__(self, full, kind):
        super().__init__()
        self.kind = kind
        if kind.startswith("convnext"):
            self.body, ln, fc = full.features, full.classifier[0], full.classifier[2]
            self.ln = ln
        elif kind == "resnet50":
            self.body = nn.Sequential(*list(full.children())[:-2])
            fc = full.fc
        elif kind == "efficientnet_b0":
            self.body, fc = full.features, full.classifier[1]
        elif kind in ("swin_t_post_norm", "swin_t_pre_norm"):
            self.feat, self.norm, fc = full.features, full.norm, full.head
        self.W = nn.Parameter(fc.weight.detach()[CLASSES].clone(), requires_grad=False)
        self.b = nn.Parameter(fc.bias.detach()[CLASSES].clone(), requires_grad=False)

    def backbone(self, x):
        if self.kind == "swin_t_post_norm":
            return self.norm(self.feat(x)).permute(0, 3, 1, 2)
        if self.kind == "swin_t_pre_norm":
            return self.feat(x).permute(0, 3, 1, 2)
        return self.body(x)

    def head(self, a):
        if self.kind == "swin_t_pre_norm":
            a = self.norm(a.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        pooled = a.mean(dim=(2, 3))
        if self.kind.startswith("convnext"):
            mu = pooled.mean(1, keepdim=True)
            s = torch.sqrt(pooled.var(1, unbiased=False, keepdim=True) + self.ln.eps)
            pooled = (pooled - mu) / s * self.ln.weight + self.ln.bias
        return pooled @ self.W.t() + self.b

    def cams(self, a):
        """Closed form; exact for pool_ln_linear and pool_linear heads (not defined for swin_t_pre_norm)."""
        hw = a.shape[2] * a.shape[3]
        pooled = a.mean(dim=(2, 3))
        if self.kind.startswith("convnext"):
            mu = pooled.mean(1, keepdim=True)
            s = torch.sqrt(pooled.var(1, unbiased=False, keepdim=True) + self.ln.eps)
            xh = ((pooled - mu) / s).unsqueeze(1)
            vb = (self.W * self.ln.weight).unsqueeze(0)
            g = (vb - vb.mean(2, keepdim=True) - xh * (vb * xh).mean(2, keepdim=True)) / s.unsqueeze(1)
        else:
            g = self.W.unsqueeze(0).expand(a.shape[0], -1, -1)
        return torch.relu(torch.einsum("bck,bkij->bcij", g / hw, a))

    def forward(self, x):
        a = self.backbone(x)
        return self.head(a), self.cams(a)


class LogitsOnly(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return self.s.head(self.s.backbone(x))


def autograd_cams(s, x):
    """Standard Grad-CAM: weights = spatial mean of d logit_c / d A, one backward per class."""
    with torch.no_grad():
        a0 = s.backbone(x)
    a = a0.clone().requires_grad_(True)
    lg = s.head(a)
    out = []
    for c in range(3):
        g, = torch.autograd.grad(lg[:, c].sum(), a, retain_graph=c < 2)
        out.append(torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1)).detach())
    grad_spread = float((g - g.mean(dim=(2, 3), keepdim=True)).abs().max() / (g.abs().max() + 1e-12))
    return torch.stack(out, 1).numpy(), a0, grad_spread


def norm(m):
    return m / (m.reshape(len(m), 3, -1).max(-1)[..., None, None] + 1e-8)


def maxdiff(a, b):
    nm = np.abs(norm(a) - norm(b)).reshape(len(a), -1).max(1)
    return {"normalised_maxabs": float(nm.max()), "normalised_maxabs_median": float(np.median(nm))}


def fix_ir(p):
    m = onnx.load(str(p))
    if m.ir_version > MAX_IR:
        m.ir_version = MAX_IR
        onnx.save(m, str(p))


def export(mod, path, outs):
    torch.onnx.export(mod, XT[:1], str(path), input_names=["image"], output_names=outs,
                      dynamic_axes={"image": {0: "batch"}, **{o: {0: "batch"} for o in outs}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    fix_ir(path)


def session(p):
    o = ort.SessionOptions()
    o.intra_op_num_threads = o.inter_op_num_threads = 1
    o.enable_cpu_mem_arena = False
    return ort.InferenceSession(str(p), sess_options=o, providers=["CPUExecutionProvider"])


def timed(sess, outs):
    x = X[:1]
    for _ in range(3):
        sess.run(outs, {"image": x})
    ts = []
    for _ in range(N_TIME):
        t = time.perf_counter()
        sess.run(outs, {"image": x})
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def head_nodes(model):
    """Every node downstream of the Grad-CAM layer (the Einsum's activation input), kept in FP32 as in the paper."""
    ein = [n for n in model.graph.node if n.op_type == "Einsum"]
    assert len(ein) == 1, f"expected one Einsum, found {len(ein)}"
    cons = {}
    for n in model.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    seen, todo = {}, [ein[0].input[1]]
    while todo:
        for n in cons.get(todo.pop(), []):
            if n.name not in seen:
                seen[n.name] = n
                todo += list(n.output)
    return sorted(seen)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


SPECS = [("convnext_tiny", "convnext_tiny", "ConvNeXt_Tiny_Weights"),
         ("convnext_small", "convnext_small", "ConvNeXt_Small_Weights"),
         ("convnext_base", "convnext_base", "ConvNeXt_Base_Weights"),
         ("resnet50", "resnet50", "ResNet50_Weights"),
         ("efficientnet_b0", "efficientnet_b0", "EfficientNet_B0_Weights"),
         ("swin_t_post_norm", "swin_t", "Swin_T_Weights"),
         ("swin_t_pre_norm", "swin_t", "Swin_T_Weights")]
FULL = {}
for kind, ctor, wname in SPECS:
    if kind not in ONLY:
        continue
    r = R["models"][kind] = {"torchvision": f"{ctor}({wname}.IMAGENET1K_V1)"}
    t0 = time.time()
    try:
        if ctor not in FULL:
            FULL.clear()
            FULL[ctor] = getattr(models, ctor)(weights=getattr(models, wname).IMAGENET1K_V1).eval()
        full = FULL[ctor]
        s = Split(full, kind).eval()
        with torch.no_grad():
            ref = full(XT)[:, CLASSES]
            lg, cf = s(XT)
        r["head_vs_full_model_logits_maxabs"] = float((ref - lg).abs().max())
        ag, a0, spread = autograd_cams(s, XT)
        r["cam_layer_shape"] = list(a0.shape[1:])
        r["autograd_gradient_position_spread_rel"] = spread
        if kind == "swin_t_pre_norm":
            post = Split(full, "swin_t_post_norm").eval()
            with torch.no_grad():
                _, cf_post = post(XT)
            d = maxdiff(ag, cf_post.numpy())
            r["closed_form_applicable"] = False
            r["autograd_pre_norm_vs_closed_form_post_norm"] = d
            na, nb = norm(ag).reshape(N_IMG * 3, -1), norm(cf_post.numpy()).reshape(N_IMG * 3, -1)
            r["pearson_r_median"] = float(np.median([np.corrcoef(p, q)[0, 1] for p, q in zip(na, nb)
                                                     if p.std() > 0 and q.std() > 0]))
            log(f"{kind}: gradient spread {spread:.3f}; autograd (pre-norm) vs closed form (post-norm) {d}")
            ROWS.append({"model": kind, "closed_form": "not applicable", "gradient_spread": spread, **d})
            save()
            continue
        r["closed_form_applicable"] = True
        r["torch_closed_form_vs_autograd"] = maxdiff(ag, cf.numpy())
        f_cam, f_log = WORK / f"{kind}_cam.onnx", WORK / f"{kind}_logits.onnx"
        export(s, f_cam, ["logits", "gradcam"])
        export(LogitsOnly(s).eval(), f_log, ["logits"])
        r["onnx_mb"] = round(f_cam.stat().st_size / 1e6, 1)
        sc, sl = session(f_cam), session(f_log)
        ol, oc = sc.run(["logits", "gradcam"], {"image": X})
        r["onnx_vs_torch_logits_maxabs"] = float(np.abs(ol - lg.numpy()).max())
        r["onnx_closed_form_vs_autograd"] = maxdiff(ag, oc)
        tl, tc = timed(sl, ["logits"]), timed(sc, ["logits", "gradcam"])
        r["ort_1thread_median_s"] = {"logits_only": tl, "logits_and_gradcam": tc, "extra_share": tc / tl - 1}
        log(f"{kind}: logits {r['head_vs_full_model_logits_maxabs']:.1e}; ONNX closed form vs autograd "
            f"{r['onnx_closed_form_vs_autograd']}; time {tl:.3f} -> {tc:.3f} s ({tc / tl - 1:+.1%})")
        save()

        # INT8-dynamic: the paper's recipe, then per-request vs batched
        pre, dyn = WORK / f"{kind}_pre.onnx", WORK / f"{kind}_int8dyn.onnx"
        try:
            quant_pre_process(str(f_cam), str(pre))
            prep = "full"
        except Exception as e1:
            log(f"{kind}: quant_pre_process failed ({e1}); retrying with skip_symbolic_shape=True")
            try:
                quant_pre_process(str(f_cam), str(pre), skip_symbolic_shape=True)
                prep = "skip_symbolic_shape"
            except Exception as e2:
                log(f"{kind}: pre-processing failed again ({e2}); quantizing the exported graph directly")
                pre.write_bytes(f_cam.read_bytes())
                prep = "none"
        R["models"][kind]["quant_pre_process"] = prep
        fix_ir(pre)
        pm = onnx.load(str(pre))
        excl = head_nodes(pm)
        n_mm = sum(n.op_type == "MatMul" and n.name not in set(excl) for n in pm.graph.node)
        quantize_dynamic(str(pre), str(dyn), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                         nodes_to_exclude=excl)
        fix_ir(dyn)
        qm = onnx.load(str(dyn))
        r["int8_dynamic"] = {"matmul_nodes_eligible": int(n_mm),
                             "dynamic_quantize_nodes": int(sum(n.op_type == "DynamicQuantizeLinear" for n in qm.graph.node)),
                             "mb": round(dyn.stat().st_size / 1e6, 1)}
        sd = session(dyn)
        per = [sd.run(["logits", "gradcam"], {"image": X[i:i + 1]}) for i in range(N_IMG)]
        pl, pc = np.concatenate([p[0] for p in per]), np.concatenate([p[1] for p in per])
        bt = [sd.run(["logits", "gradcam"], {"image": X[i:i + BATCH]}) for i in range(0, N_IMG, BATCH)]
        bl, bc = np.concatenate([b[0] for b in bt]), np.concatenate([b[1] for b in bt])
        pp, bp, fp = softmax(pl), softmax(bl), softmax(ol)
        rel = np.abs(pc - bc).reshape(N_IMG, -1).max(1) / (np.abs(pc).reshape(N_IMG, -1).max(1) + 1e-12)
        r["int8_dynamic"].update({
            "batched_vs_per_request_prob_maxabs": float(np.abs(pp - bp).max()),
            "batched_vs_per_request_logit_maxabs": float(np.abs(pl - bl).max()),
            "batched_vs_per_request_top1_changes": int((pp.argmax(1) != bp.argmax(1)).sum()),
            "batched_vs_per_request_map_rel_diff": {"median": float(np.median(rel)), "max": float(rel.max())},
            "per_request_vs_fp32_prob_maxabs": float(np.abs(pp - fp).max()),
            "per_request_vs_fp32_top1_changes": int((pp.argmax(1) != fp.argmax(1)).sum()),
            "batch_dependent": bool(np.abs(pl - bl).max() > 1e-5)})
        log(f"{kind} INT8-dyn: {json.dumps(r['int8_dynamic'])}")
        ROWS.append({"model": kind, "closed_form": "exact", "gradient_spread": spread,
                     **r["onnx_closed_form_vs_autograd"], "time_logits_s": tl, "time_cam_s": tc,
                     "extra_share": tc / tl - 1, **{f"int8dyn_{k}": v for k, v in r["int8_dynamic"].items()
                                                    if not isinstance(v, dict)},
                     "int8dyn_map_rel_diff_max": r["int8_dynamic"]["batched_vs_per_request_map_rel_diff"]["max"]})
        for p in (f_cam, f_log, pre, dyn):
            p.unlink(missing_ok=True)
    except Exception as e:  # record and continue with the next model
        r["error"] = f"{type(e).__name__}: {e}"
        r["traceback"] = traceback.format_exc()[-1500:]
        log(f"{kind}: FAILED {r['error']}")
    r["seconds"] = round(time.time() - t0, 1)
    save()
log("Done.")
