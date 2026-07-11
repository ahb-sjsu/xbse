"""SIMULATED shadow-pilot dry-run (prereg_shadow_pilot.md) — NRP LLMs as simulated adjudicators.

Incumbent adjudicator = qwen3 (blinded to engine scores); double-adjudication audit = glm-5 (-> kappa).
Traffic = Measuring Hate Speech (text + hate_speech_score oracle + identity slices). Engine = xbse
validated-axis violation score (commonsense-valence via care_joint + physical_harm if available).

CAVEATS printed on every run: (1) LLM-simulated adjudicators are a REHEARSAL of the C1-C5 measurement
pipeline, not evidence about real deployment; (2) engine and LLM judge may share a harm notion, which
can INFLATE C1 (triage value). A high C1 here is a pipeline check, not a product claim.
"""

import json
import os
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.environ.get("XBSE_SRC", "/home/claude/xbse/src"))
os.environ.setdefault("HF_HOME", "/home/claude/hf_cache")

import torch  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from xbse.encoder import BSEEncoder  # noqa: E402
from xbse.instances.joint_builders import (  # noqa: E402
    FOUNDATIONS,
    MS_CARE,
    _signed_jsonl_simple,
    _social_chem_rows,
)

CKDIR = "/home/claude/xbse_ckpt"
TOKEN = open("/home/claude/.llmtoken").read().strip()
URL = "https://ellm.nrp-nautilus.io/v1/chat/completions"
N = int(os.environ.get("SIM_N", "360"))
WEEKS = 8
K_FRAC = 0.20            # daily review capacity = top 20%
OP_THRESH_Q = 0.80       # operating point: flag top 20% by engine score
RATIO_C1 = 1.15
rng = np.random.default_rng(0)

ADJ = ("You are a content moderator. Does this user post violate a policy against hateful or harmful "
       "content targeting a person or group? Answer with ONLY 'YES' or 'NO'.\n\nPost: {t}\n\nAnswer:")


# NRP models are reasoning models: the YES/NO lands in `content` but only if max_tokens is large
# enough for the thinking to finish first (max_tokens=4 truncates -> content=null). Use non-reasoning-
# fast models (gemma-small, kimi) and parse content first, reasoning as fallback.
INC_MODEL = os.environ.get("SIM_INC_MODEL", "gemma-small")
DBL_MODEL = os.environ.get("SIM_DBL_MODEL", "kimi")


def llm_msg(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 300}).encode()
    req = urllib.request.Request(URL, body,
                                 {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())["choices"][0]["message"]
        except Exception:
            time.sleep(3)
    return {}


def adjudicate(model, text):
    m = llm_msg(model, ADJ.format(t=text[:700]))
    for field in ((m.get("content") or ""), (m.get("reasoning") or "")):
        u = field.upper()
        y, n = u.rfind("YES"), u.rfind("NO")
        if y >= 0 or n >= 0:
            return 1 if y > n else 0
    return 0


# ---------- traffic ----------
from datasets import load_dataset  # noqa: E402
ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
byc = {}
for r in ds:
    cid = r["comment_id"]
    if cid not in byc:
        byc[cid] = {"text": r["text"], "sc": [], "race": r.get("target_race"),
                    "gender": r.get("target_gender"), "religion": r.get("target_religion")}
    byc[cid]["sc"].append(float(r["hate_speech_score"]))
allc = [v for v in byc.values() if len(v["text"]) >= 12]
pick = rng.choice(len(allc), min(N, len(allc)), replace=False)
items = []
for i in pick:
    v = allc[i]
    items.append({"text": v["text"], "oracle": 1 if np.mean(v["sc"]) > 0.5 else 0,
                  "race": v["race"], "gender": v["gender"], "religion": v["religion"]})
N = len(items)
print(f"[sim] traffic N={N} oracle-harm rate={np.mean([it['oracle'] for it in items]):.2f}", flush=True)


# ---------- engine (validated-axis violation score) ----------
def build_axis(enc, pos, neg):
    zp = enc.encode(pos[:400]).mean(0)
    zn = enc.encode(neg[:400]).mean(0)
    a = zp - zn
    return a / a.norm()


enc = BSEEncoder(base_model="BAAI/bge-m3", pooling="mean", max_len=128, device="cuda")
enc.load_state_dict(torch.load(f"{CKDIR}/care_joint.pt", map_location="cuda"))
enc.eval()
# valence axis from care builder exemplars (+ = upheld, - = violation)
sc_rows = _social_chem_rows("care-harm", FOUNDATIONS["care"][1]) + \
    [(t, s) for t, s in _signed_jsonl_simple(MS_CARE)]
pos = [t for t, s in sc_rows if s == "+"]
neg = [t for t, s in sc_rows if s == "-"]
axis = build_axis(enc, pos, neg)
texts = [it["text"] for it in items]
z = enc.encode(texts)
val_proj = (z * axis).sum(-1).cpu().numpy()   # high = upheld; low = violation
harm_score = -val_proj                         # high = more violation/harm  (engine queue score)
for it, hs in zip(items, harm_score):
    it["engine"] = float(hs)
print("[sim] engine scored via commonsense-valence axis (violation direction)", flush=True)

# ---------- incumbent + double adjudication (NRP LLMs) ----------
print(f"[sim] incumbent adjudication ({INC_MODEL})...", flush=True)
for j, it in enumerate(items):
    it["inc"] = adjudicate(INC_MODEL, it["text"])
    if (j + 1) % 60 == 0:
        print(f"   {j+1}/{N}", flush=True)
sub = rng.choice(N, max(20, N // 5), replace=False)
print(f"[sim] double-adjudication audit ({DBL_MODEL}) on {len(sub)} items...", flush=True)
dbl = {int(i): adjudicate(DBL_MODEL, items[i]["text"]) for i in sub}
kappa = cohen_kappa_score([items[i]["inc"] for i in sub], [dbl[i] for i in sub])
print(f"[sim] incumbent<->audit Cohen kappa = {kappa:.3f}", flush=True)

# oracle check: how well does the simulated incumbent track the MHS human oracle?
inc = np.array([it["inc"] for it in items]); orc = np.array([it["oracle"] for it in items])
print(f"[sim] incumbent-vs-oracle agreement = {(inc == orc).mean():.3f} "
      f"(incumbent harm rate {inc.mean():.2f}, oracle {orc.mean():.2f})", flush=True)


# ---------- C1: weekly precision@k, engine vs chronological ----------
def prec_at_k(order_idx, labels, k):
    top = order_idx[:k]
    return labels[top].mean()


order_chrono = np.arange(N)
rng.shuffle(order_chrono)  # 'chronological' = arrival order (random here)
week_bins = np.array_split(order_chrono, WEEKS)
c1_pass_weeks = 0
print("\n=== C1 weekly (precision@k, engine vs chronological) ===", flush=True)
for w, wk in enumerate(week_bins):
    lab = inc[wk]
    k = max(1, int(len(wk) * K_FRAC))
    eng_order = wk[np.argsort(-harm_score[wk])]
    p_eng = inc[eng_order[:k]].mean()
    p_base = lab[:k].mean() if lab[:k].size else 0.0   # chronological top-k
    # bootstrap CI on the ratio (resample items in the week)
    ratios = []
    for _ in range(500):
        bs = rng.choice(len(wk), len(wk), replace=True)
        wkl = inc[wk[bs]]
        eo = wk[bs][np.argsort(-harm_score[wk[bs]])]
        pe = inc[eo[:k]].mean()
        pb = wkl[:k].mean()
        if pb > 0:
            ratios.append(pe / pb)
    lo = np.percentile(ratios, 2.5) if ratios else float("nan")
    ok = lo >= RATIO_C1
    c1_pass_weeks += ok
    print(f"  wk{w}: p@k eng={p_eng:.2f} base={p_base:.2f} ratio_CI_lo={lo:.2f} "
          f"{'PASS' if ok else 'fail'}", flush=True)
C1 = c1_pass_weeks >= 6
print(f"C1: {c1_pass_weeks}/{WEEKS} weeks -> {'PASS' if C1 else 'FAIL'}", flush=True)

# ---------- C3: identity-slice benign FPR ratio ----------
thr = np.quantile(harm_score, OP_THRESH_Q)
flag = harm_score >= thr
benign = inc == 0
overall_fpr = flag[benign].mean() if benign.any() else 0.0
print(f"\n=== C3 slice benign-FPR (overall {overall_fpr:.2f}, op-thresh q{OP_THRESH_Q}) ===", flush=True)
c3_viol = []
for field in ("race", "gender", "religion"):
    groups = defaultdict(list)
    for i, it in enumerate(items):
        if it[field]:
            groups[str(it[field]).split(",")[0]].append(i)
    for g, idxs in groups.items():
        idxs = [i for i in idxs if benign[i]]
        if len(idxs) < 10:
            continue
        fpr = flag[idxs].mean()
        ratio = fpr / overall_fpr if overall_fpr > 0 else float("nan")
        bad = ratio > 1.25
        if bad:
            c3_viol.append((field, g, ratio))
        print(f"  {field}:{g:<14} n={len(idxs):3d} FPR={fpr:.2f} ratio={ratio:.2f}"
              f"{'  <-- >1.25' if bad else ''}", flush=True)
C3 = len(c3_viol) == 0
print(f"C3: {'PASS' if C3 else 'FAIL (' + str(len(c3_viol)) + ' slice-violations)'}", flush=True)

# ---------- C5: missed-harm audit (low-score items) ----------
low = harm_score < np.quantile(harm_score, 0.5)
missed = inc[low].mean()   # fraction of low-score items the incumbent calls harmful (false negatives)
print(f"\n=== C5 missed-harm (incumbent-harmful among engine low-score) = {missed:.2f} ===", flush=True)

print("\n=== SIMULATED VERDICT (methods rehearsal — NOT a deployment result) ===", flush=True)
print(f"  kappa(incumbent,audit)={kappa:.3f}  C1={'PASS' if C1 else 'FAIL'}({c1_pass_weeks}/8)  "
      f"C3={'PASS' if C3 else 'FAIL'}  C5 missed-harm={missed:.2f}", flush=True)
verdict = "PASS" if (C1 and C3 and kappa >= 0.4) else ("INCONCLUSIVE" if kappa < 0.4 else "FAIL")
print(f"  -> {verdict}  (INCONCLUSIVE if kappa<0.4: reference standard too noisy)", flush=True)
print("  CAVEAT: qwen3/glm-5 simulate humans; engine/judge shared harm notion can inflate C1.", flush=True)
