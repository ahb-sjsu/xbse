"""discover.py — structural-fuzz for INSTANCE DISCOVERY.

An instance is a tuple (domain, invariance-axis, structure-axis, independent-label). We already
built the executable admission filter as a verifier. This closes the loop into a search:

    candidate PairSources  ->  admission filter (throws on circularity / missing label)
                           ->  CHEAP validation proxy (frozen-encoder structure-vs-surface AUROC)
                           ->  ranked shortlist -> build only the top-ranked ones.

The proxy is structure-vs-surface AUROC under a FROZEN base encoder (no training). It's a *screen*,
not a verdict, and it's asymmetric:
  - HIGH frozen proxy  => the structure is already linearly present in the base => will validate easily.
  - LOW  frozen proxy  => structure is orthogonal-to / adversarial-with the base (like MoBSE's 0.20);
                          needs training and may hit a domain ceiling — invest with eyes open.
So: reject anything the admission filter throws on; among the admissible, prefer high frozen proxy,
but a low proxy is "needs training + risk", not "impossible" (MoBSE was 0.20 frozen -> 0.77 trained).
"""
from __future__ import annotations

from .encoder import BSEEncoder
from .validate import structure_vs_surface_auroc, surface_invariance
from .admission import AdmissionError


def frozen_proxy(source, encoder) -> dict:
    ev = source.heldout_eval()
    return {
        "structure_auroc_frozen": structure_vs_surface_auroc(encoder, ev["structural_pairs"]),
        "surface_inv_frozen": surface_invariance(encoder, ev["surface_pairs"]),
    }


def discover(sources, base_model: str = "BAAI/bge-m3", device: str = "cuda", max_len: int = 128):
    """Admission-check + frozen-proxy each candidate PairSource; return a ranked table.

    Returns list of dicts: {name, admissible, structure_auroc_frozen, surface_inv_frozen, note}.
    """
    enc = BSEEncoder(base_model=base_model, max_len=max_len, device=device)
    rows = []
    for s in sources:
        try:
            s.check_admission()
        except AdmissionError as e:
            rows.append({"name": s.name, "admissible": False, "structure_auroc_frozen": None,
                         "surface_inv_frozen": None, "note": f"REJECTED: {str(e)[:80]}"})
            continue
        try:
            p = frozen_proxy(s, enc)
            score = p["structure_auroc_frozen"]
            note = ("high — validates easily" if score >= 0.75 else
                    "low — needs training + risk" if score <= 0.55 else "moderate")
            rows.append({"name": s.name, "admissible": True, **p, "note": note})
        except Exception as e:
            rows.append({"name": s.name, "admissible": True, "structure_auroc_frozen": None,
                         "surface_inv_frozen": None, "note": f"proxy error: {str(e)[:60]}"})
    rows.sort(key=lambda r: (r["admissible"], r["structure_auroc_frozen"] or -1), reverse=True)
    return rows


def print_table(rows) -> None:
    print(f"{'instance':16s} {'admit':6s} {'frozen-AUROC':12s} {'note'}")
    for r in rows:
        au = f"{r['structure_auroc_frozen']:.3f}" if r["structure_auroc_frozen"] is not None else "  -  "
        print(f"  {r['name']:14s} {str(r['admissible']):6s} {au:12s} {r['note']}")
