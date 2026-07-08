# xbse — a modular framework for domain-specific BERT Sentence Embeddings

`*-BSE` models (LaBSE, LeBSE, MoBSE, …) are the *same architecture* trained to be **invariant
to one axis** and **sensitive to another**. `xbse` factors out everything they share and leaves
each domain as a small plugin — so a new `*-BSE` is a `PairSource` + a config, not a new repo,
and **every instance is validated by the same gate.**

## The idea in one table

| model | base + arch | invariance (→ positives) | sensitivity (→ negatives) |
|---|---|---|---|
| LaBSE | BERT dual-encoder | language (translation pairs) | meaning |
| LeBSE | BERT dual-encoder | surface / citation form | legal holding |
| **MoBSE** | BGE-M3 dual-encoder | wording / framing / Hohfeld gauge | moral judgment / structure |

Only the **PairSource** (invariance→positives, sensitivity→negatives), an optional **adversary**
(strip a named nuisance), and the **validation labels** differ. The encoder, the contrastive
objective, and the validation harness are written once.

## Architecture

```
xbse/
  encoder.py     BSEEncoder: base transformer -> pooling -> projection -> L2-norm   (shared)
  pairs.py       PairSource ABC yielding (anchor, positive, negative)               (shared iface)
  objective.py   InfoNCE / triplet contrastive loss                                 (shared)
  adversary.py   optional gradient-reversal head to strip a nuisance variable       (shared)
  validate.py    THE GATE: structure-vs-surface AUROC + fuzz ratio + OOD control    (shared)
  train.py       training loop wiring the above                                     (shared)
  instances/
    mobse.py     PairSource + config: positives = same-judgment+paraphrase,         (per-domain)
                 negatives = opposite-judgment  (Social-Chem-101 / Scruples / Moral-Machine)
```

## The non-negotiable discipline (read before adding code)

This project is built in **falsification order** with **hard stops**. See `docs/MOBSE_PLAN.md`.

1. **Build an instance** (e.g. MoBSE v1).
2. **Validate it** with the shared harness — structure-vs-surface AUROC on held-out data, fuzz
   ratio > 1 (surface changes move `z` little, structural changes a lot), and an out-of-domain
   control. **HARD STOP: if it fails the gate, no downstream tools, no claims — iterate the
   encoder.** (An earlier moral-encoder attempt failed here: fuzz ratio inverted to 0.20×.)
3. Only a *validated* embedding earns `encode` / `distance` / `decompose`.
4. Any geometric tool built on top (e.g. `crosses_stratum`) is a **falsifiable hypothesis about
   the geometry** run as a pre-registered test — promoted from "hypothesis" to "operation" only
   on a green light. The toolkit is simultaneously a reasoning API and a probe battery.

**Language discipline:** operations are named for what they *measure*, not for retired claims.
A harm operation computes a *representation-invariant harm ledger* (invariance, testable) — never
a "conserved quantity." A path operation reports *holonomy / path-dependence* — nonzero or not,
full stop.

## Why the shared gate matters

Because MoBSE cannot grade itself easier than LeBSE did. One validation harness → one comparable
number (structure-vs-surface AUROC, fuzz ratio) across the whole family. Validation is a property
of the framework, not a favor each instance does itself.

## Status

Scaffold. First instance: **MoBSE v1** — not yet trained, not yet validated. Nothing downstream
of the validation gate exists on purpose.

## Kinship (not merged, on purpose)

`*-BSE` learns **invariance** — it quotients a nuisance group out. Its equivariant cousins live
elsewhere and stay separate: [`latte`](https://github.com/ahb-sjsu/latte) and
[`arc-equivariant-search`](https://github.com/ahb-sjsu/arc-equivariant-search) *covary* with a
symmetry group (D4×color augment → solve → vote = averaging over the orbit). Same principle — a
transformation group acts; you either mod it out (here) or transform with it (there) — different
codebase. The shared thing is the principle, not the repo.

## License

MIT © Andrew H. Bond (SJSU).
