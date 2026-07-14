"""Generate the three figures for the Moral Spectrum Analyzer paper.
Vector PDF output for xelatex. CVD-safe Okabe-Ito palette (validated).
Numbers are drawn verbatim from the verified source artifacts (see paper Appendix B).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4d4d4d",
    "axes.linewidth": 0.8,
    "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
})

BLUE = "#0072B2"     # valence / collapses-onto-G
VERM = "#D55E00"     # presence / separable
GRAY = "#8a8a8a"
CELLBG = "#f4f6f8"   # neutral cell fill
LINE = "#c9d0d8"

# ---------------------------------------------------------------- Fig 0: the MoralVector structure
def fig_moralvector(path):
    """The frozen 3x3 scope x mode ontology, each canonical dimension colored by
    its measured general-factor loading beta (blue >= 0.8 shares G, vermillion
    separable, gray unmeasured), plus the three extension channels below."""
    import matplotlib.patches as mp
    MONO = "DejaVu Sans Mono"
    # (k, key, col=mode idx, row=scope idx, beta)  rows: Individual/Relational/Collective
    dims = [
        (3, "autonomy_respect",       0, 0,  0.022),
        (1, "rights_respect",         1, 0,  None),
        (4, "privacy_protection",     2, 0,  0.251),
        (6, "virtue_care",            0, 1,  0.989),
        (0, "physical_harm",          1, 1,  0.347),
        (8, "epistemic_quality",      2, 1,  0.827),
        (2, "fairness_equity",        0, 2,  0.982),
        (7, "legitimacy_trust",       1, 2,  0.982),
        (5, "societal_environmental", 2, 2, -0.009),
    ]
    modes = [("What Matters", "values"), ("Who Decides", "deontic"), ("What We Know", "epistemic")]
    scopes = ["Individual", "Relational", "Collective"]
    ext = [("identity_attack", 0.361, "discovered §5.6"),
           ("purity", 0.957, "MFT binding"), ("loyalty", 0.975, "MFT binding")]
    # Not gate-validated (no validated second corpus) -> shown pending, in gray,
    # so an unclaimed dimension never reads as a confident channel (§5.1).
    PENDING = {"rights_respect", "societal_environmental"}

    def regime(b, key=None):
        if key in PENDING or b is None:
            return GRAY
        return BLUE if b >= 0.8 else VERM

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    W, H, G = 1.0, 1.0, 0.10          # slot size and gap
    def cell(c, r):                    # bottom-left of the drawable cell (row 0 at top)
        return c * W, (2 - r) * H

    for k, key, c, r, b in dims:
        x0, y0 = cell(c, r)
        pending = key in PENDING
        col = regime(b, key)
        ax.add_patch(mp.FancyBboxPatch((x0, y0), W - G, H - G, boxstyle="round,pad=0,rounding_size=0.04",
                                       fc=CELLBG, ec=LINE, lw=0.9, zorder=2))
        ax.add_patch(mp.Rectangle((x0, y0 + H - G - 0.05), W - G, 0.05, fc=col, ec="none", zorder=3))
        ax.text(x0 + 0.06, y0 + H - G - 0.14, f"k{k}", family=MONO, fontsize=7, color=GRAY, zorder=4)
        # key: shrink the two longest identifiers so nothing overprints the cell
        ksize = 6.6 if len(key) >= 18 else 7.4
        ax.text(x0 + (W - G) / 2, y0 + 0.52, key, family=MONO, fontsize=ksize, color="#1a1a1a",
                ha="center", va="center", zorder=4)
        # beta mini-bar + label
        bx, bw, by = x0 + 0.08, (W - G) - 0.16, y0 + 0.28
        ax.add_patch(mp.Rectangle((bx, by), bw, 0.035, fc=LINE, ec="none", zorder=3))
        if b is not None and not pending:
            ax.add_patch(mp.Rectangle((bx, by), bw * max(0.0, min(1.0, b)), 0.035, fc=col, ec="none", zorder=4))
        blabel = "β n/a" if b is None else f"β {b:.2f}"
        tag = "pending" if pending else ("shares G" if b >= 0.8 else "separable")
        ax.text(x0 + 0.08, y0 + 0.12, f"{blabel}", family=MONO, fontsize=7.4, color=col,
                va="center", ha="left", zorder=4)
        ax.text(x0 + (W - G) - 0.08, y0 + 0.12, tag, family=MONO, fontsize=6.6, color=GRAY,
                ha="right", va="center", zorder=4)

    # column headers (mode) and row labels (scope)
    for c, (m, sub) in enumerate(modes):
        cx = cell(c, 0)[0] + (W - G) / 2
        ax.text(cx, 3.15, m, fontsize=8.5, ha="center", color="#1a1a1a", weight="bold")
        ax.text(cx, 3.02, sub, family=MONO, fontsize=7.5, ha="center", color=GRAY)
    for r, s in enumerate(scopes):
        _, y0 = cell(0, r)
        ax.text(-0.20, y0 + (H - G) / 2, s, fontsize=8.5, ha="center", va="center",
                color="#1a1a1a", rotation=90)
    # axis captions, pushed clear of the headers
    ax.text(-0.60, (2 - 1) * H + (H - G) / 2, "SCOPE", family=MONO, fontsize=7.5, color=GRAY,
            ha="center", va="center", rotation=90)
    ax.text(cell(1, 0)[0] + (W - G) / 2, 3.40, "MODE  →", family=MONO, fontsize=7.5,
            color=GRAY, ha="center")

    # extension-channel band
    by0 = -1.15
    ax.text(0.0, by0 + 0.72, "EXTENSION CHANNELS — ride outside the frozen nine (metadata, not the k-axis)",
            family=MONO, fontsize=7.2, color=GRAY)
    for i, (key, b, note) in enumerate(ext):
        x0 = i * W
        col = regime(b)
        ax.add_patch(mp.FancyBboxPatch((x0, by0), W - G, 0.56, boxstyle="round,pad=0,rounding_size=0.04",
                                       fc=CELLBG, ec=LINE, lw=0.9, zorder=2))
        ax.add_patch(mp.Rectangle((x0, by0), 0.045, 0.56, fc=col, ec="none", zorder=3))
        ksize = 6.8 if len(key) >= 14 else 7.6
        ax.text(x0 + 0.12, by0 + 0.37, key, family=MONO, fontsize=ksize, color="#1a1a1a", va="center", zorder=4)
        ax.text(x0 + 0.12, by0 + 0.15, note, family=MONO, fontsize=6.4, color=GRAY, va="center", zorder=4)
        ax.text(x0 + (W - G) - 0.06, by0 + 0.37, f"β {b:.2f}", family=MONO, fontsize=7.4, color=col,
                ha="right", va="center", zorder=4)
        ax.text(x0 + (W - G) - 0.06, by0 + 0.15, "shares G" if b >= 0.8 else "separable", family=MONO,
                fontsize=6.4, color=GRAY, ha="right", va="center", zorder=4)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color=BLUE, lw=0, ms=8, label="shares general factor G (β ≥ 0.8)"),
        Line2D([0], [0], marker="s", color=VERM, lw=0, ms=8, label="separable subspace (β < 0.8)"),
        Line2D([0], [0], marker="s", color=GRAY, lw=0, ms=8, label="pending — not gate-validated"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=7.2, handletextpad=0.4, columnspacing=1.4)

    ax.set_xlim(-0.82, 3.05)
    ax.set_ylim(by0 - 0.55, 3.55)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 1: cross-dataset collapse
def fig_crossdataset(path):
    # dimension: (in-domain inflated, cross-dataset honest)  — verified §5.1 table
    data = [
        ("privacy_protection", 0.865, 0.853),
        ("epistemic_quality",  0.90,  0.817),
        ("virtue_care",        0.85,  0.811),
        ("fairness_equity",    0.87,  0.789),
        ("autonomy_respect",   0.955, 0.747),
        ("legitimacy_trust",   0.80,  0.708),
        ("physical_harm",      0.88,  0.622),
    ]
    data.sort(key=lambda r: r[2])  # by cross-dataset
    names = [r[0] for r in data]
    indom = [r[1] for r in data]
    cross = [r[2] for r in data]
    y = list(range(len(names)))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.barh([v + h/2 for v in y], indom, height=h, color=GRAY, alpha=0.55,
            label="in-domain (inflated)", zorder=3)
    ax.barh([v - h/2 for v in y], cross, height=h, color=BLUE,
            label="cross-dataset (honest)", zorder=3)
    for v, x in zip([v + h/2 for v in y], indom):
        ax.text(x + 0.006, v, f"{x:.2f}", va="center", ha="left", fontsize=7, color=GRAY)
    for v, x in zip([v - h/2 for v in y], cross):
        ax.text(x + 0.006, v, f"{x:.2f}", va="center", ha="left", fontsize=7, color=BLUE)
    ax.axvline(0.5, color=GRAY, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(0.5, len(names) - 0.35, " chance", color=GRAY, fontsize=7, va="center")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.45, 1.0); ax.set_xlabel("held-out AUROC")
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("In-domain accuracy overstates cross-dataset transfer",
                 fontsize=9.5, loc="left", pad=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: valence/identity dissociation
def fig_dissociation(path):
    # (channel, kind, lam0, lam1, passes_at_lam1)  — verified §5.7 table
    rows = [
        ("loyalty (val)",   "valence",  0.911, 0.892, True),
        ("G (val)",         "valence",  0.856, 0.850, True),
        ("purity (val)",    "valence",  0.811, 0.801, True),
        ("purity (pres)",   "presence", 0.719, 0.699, True),
        ("loyalty (pres)",  "presence", 0.661, 0.641, False),
        ("legitimacy (pres)","presence",0.585, 0.544, False),
        ("care (pres)",     "presence", 0.574, 0.585, False),
        ("fairness (pres)", "presence", 0.569, 0.499, False),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x0, x1 = 0, 1
    # left-label y nudges to de-crowd the three clustered failing presence channels
    left_dy = {"legitimacy (pres)": 0.016, "care (pres)": 0.0, "fairness (pres)": -0.016}
    right_dy = {"legitimacy (pres)": 0.014, "care (pres)": 0.0, "fairness (pres)": -0.014}
    for name, kind, a, b, ok in rows:
        c = BLUE if kind == "valence" else VERM
        ax.plot([x0, x1], [a, b], color=c, lw=1.6, zorder=3, alpha=0.9)
        ax.plot(x0, a, "o", color=c, ms=5, zorder=4)
        # filled = passes gate at lam=1, open = fails
        if ok:
            ax.plot(x1, b, "o", color=c, ms=6, zorder=4)
        else:
            ax.plot(x1, b, "o", color="white", ms=6, mec=c, mew=1.6, zorder=4)
        ax.text(x0 - 0.03, a + left_dy.get(name, 0.0), name, va="center", ha="right",
                fontsize=7.2, color=c)
        ax.text(x1 + 0.03, b + right_dy.get(name, 0.0), f"{b:.2f}", va="center", ha="left",
                fontsize=7.2, color=c)
    ax.set_xlim(-0.55, 1.35); ax.set_ylim(0.45, 0.95)
    ax.set_xticks([x0, x1]); ax.set_xticklabels(["adversary off (λ=0)", "adversary on (λ=1)"])
    ax.set_ylabel("cross-dataset AUROC")
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.7, zorder=0); ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    handles = [
        Line2D([0],[0], color=BLUE, lw=1.6, label="valence channel"),
        Line2D([0],[0], color=VERM, lw=1.6, label="presence channel"),
        Line2D([0],[0], marker="o", color="#4d4d4d", lw=0, ms=6, label="passes gate"),
        Line2D([0],[0], marker="o", color="white", mec="#4d4d4d", mew=1.6, lw=0, ms=6, label="fails gate"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.2, ncol=2)
    ax.set_title("Valence transfers regardless of the adversary; identity does not",
                 fontsize=9.5, loc="left", pad=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: bifactor beta loadings
def fig_bifactor(path):
    # (axis, beta on G)  — verified §5.3 (bifactor_A2_result.json)
    rows = [
        ("care", 0.989), ("fairness", 0.982), ("legitimacy", 0.982),
        ("loyalty", 0.975), ("purity", 0.957), ("epistemic", 0.827),
        ("identity_attack", 0.361), ("physical_harm", 0.347),
        ("privacy", 0.251), ("autonomy", 0.022), ("societal_env", -0.009),
    ]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]; betas = [r[1] for r in rows]
    # regime by loading: >=0.8 collapses onto G (blue), else separable (vermillion)
    colors = [BLUE if b >= 0.8 else VERM for b in betas]
    y = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(y, betas, color=colors, height=0.62, zorder=3)
    for v, b in zip(y, betas):
        ax.text(b + (0.012 if b >= 0 else -0.012), v, f"{b:.2f}",
                va="center", ha="left" if b >= 0 else "right", fontsize=7,
                color="#1a1a1a")
    ax.axvline(0, color="#4d4d4d", lw=0.8, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(-0.15, 1.12); ax.set_xlabel("loading β on the general factor G")
    ax.xaxis.grid(True, color="#e6e6e6", lw=0.7, zorder=0); ax.set_axisbelow(True)
    handles = [
        Line2D([0],[0], marker="s", color=BLUE, lw=0, ms=8, label="collapses onto G (β ≥ 0.8)"),
        Line2D([0],[0], marker="s", color=VERM, lw=0, ms=8, label="separable (β < 0.8)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title("Two regimes: a shared-corpus family collapses onto G; independent-corpus axes stay separable",
                 fontsize=8.6, loc="left", pad=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    fig_moralvector(os.path.join(d, "fig0_moralvector.pdf"))
    fig_crossdataset(os.path.join(d, "fig1_crossdataset.pdf"))
    fig_dissociation(os.path.join(d, "fig2_dissociation.pdf"))
    fig_bifactor(os.path.join(d, "fig3_bifactor.pdf"))
    print("wrote fig0_moralvector.pdf, fig1_crossdataset.pdf, fig2_dissociation.pdf, fig3_bifactor.pdf")
