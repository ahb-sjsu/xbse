# NRP Nautilus — A100 quota request (namespace `<namespace>`)

*Send via the NRP Matrix support channel (`#support:matrix.nrp-nautilus.io`) or
`support@nationalresearchplatform.org`. Copy the body below.*

---

**Subject:** A100 quota request for namespace `<namespace>` — full fine-tuning of a 7B embedding model

**Requester:** Andrew H. Bond, San José State University · `agi.hpc@gmail.com`
**Namespace:** `<namespace>`

**Current quota (verified via `kubectl describe resourcequota`):**
`nvidia.com/a100` = **0/0** (requests and limits). A100/H100/H200/GH200 are all quota-banned to 0 for
our namespace; we currently schedule only the shared Ampere/Ada pool (RTX-3090/A10/L40/L40S/A6000,
≤48 GB).

**Requested change:** raise `nvidia.com/a100` to **2** (requests and limits) for a time-limited
research window (~2 weeks). If the SXM4-80GB pool is gated under a distinct resource name, please
grant whichever key maps to an **80 GB A100**. We will accept **1** if 2 is not available.

**Why we need it (and why ≤48 GB does not suffice).** We are validating a moral-dimension
sentence-embedding framework — per-dimension encoders that score a text's moral valence (harm,
fairness, privacy, honesty, rights, …), used as the perception layer of an AI-ethics engine. A
reviewer of the accompanying paper asked for an **encoder-invariance control**: does our central
result (cross-dataset transferability varies systematically by dimension) hold with a stronger
encoder, or is it an artifact of the 560M-parameter BGE-M3 baseline? Answering this cleanly requires
**full fine-tuning of a 7B embedder** (`Alibaba-NLP/gte-Qwen2-7B-instruct` / `Qwen3-Embedding-8B`)
under the **same training recipe** as the baseline. Full fine-tuning of a 7B model needs ~110 GB of
optimizer state (fp32 AdamW moments + master weights), which does not fit the ≤48 GB cards we can
currently schedule; those force QLoRA, which changes the recipe and **confounds the very invariance
comparison** the experiment is designed to make. A single 80 GB A100 (with gradient checkpointing +
8-bit Adam) makes the clean full-fine-tune feasible.

**Usage pattern (good-citizen commitment).** Time-limited, high-utilization **batch Jobs** only
(`kind: Job`, `restartPolicy: Never`, `ttlSecondsAfterFinished` set): **~2–4 runs, each ~1–2 h at
sustained >90% GPU utilization**, resources released immediately on completion. **No interactive
sessions, no idle GPU holds.** We already run policy-clean batch fine-tunes on the shared Ampere/Ada
pool (e.g. the `arc-latte-base-ft` Job) and will follow the same discipline. The quota may be
reduced back to 0 after the experiment window.

**Timeline:** the experiment is ready to run; a 2-week window is sufficient.

Thank you for maintaining Nautilus — happy to provide any further detail.

---

## If the request is declined or pending

No blocker for the science — we proceed on the schedulable ≤48 GB pool with two runs that need **no
permission**:

| run | encoder | recipe | GPU (schedulable now) | comparison role |
|---|---|---|---|---|
| primary | `gte-Qwen2-1.5B-instruct` | **full fine-tune** | L40 / L40S / A6000 / A40 (48 GB) | clean invariance test (= BGE-M3 recipe) |
| scale | `gte-Qwen2-7B-instruct` | QLoRA (reuses `base-ft.job.yaml`) | A10 / RTX-3090 / L4 (24 GB) | scale robustness (LoRA caveat noted) |

The A100 quota only unlocks the *airtight* 7B **full**-fine-tune; the 1.5B-full-FT already gives a
valid encoder-invariance result without it.
