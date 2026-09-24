# Decisions

Every decision that shaped this project, in order, with who made it and why.
**Direction** — the research question, framing and interpretation — is Moinuddin's.
**Execution** — code, measurement, fixes — is Claude's, and anything that could change a
conclusion is surfaced rather than decided silently.

| # | Date | Decision | Who | Why | Alternatives set aside |
|---|---|---|---|---|---|
| 1 | 2026-09-24 | Build a project **on top of** Laya rather than put someone else's model on the portfolio | Moinuddin | A portfolio must show his own work; laya-mlx is another author's port | Embed Laya in the site's Lab as a game |
| 2 | 2026-09-24 | Frame it as **"is this hype?"** — Laya/Jev are a cross-encoder scoring options with a softmax, an idea that predates them | Moinuddin | "It takes the same as a transformer but doesn't write text, it gives softmax; we had this before" | A confidence-gated cascade (Laya → LLM) product |
| 3 | 2026-09-24 | Research both vendors' claims **before** designing anything | Moinuddin | Ground the framing in what they actually publish | Start building immediately |
| 4 | 2026-09-24 | Compare against **the classifiers we already had**, not LLMs: NLI zero-shot, embeddings + LR, SetFit, (later) fine-tuned ModernBERT | Direction: Moinuddin · design: Claude | Jev compares only with LLMs, Laya only with Jev; nobody tested the obvious baselines. `nibzard/decision-model-benchmark` covers Jev vs LLMs, so this gap is still open | Another LLM comparison |
| 5 | 2026-09-24 | Fixed test (1,000), calibration (500) and k-shot splits (8/16/64 per class × 5 seeds), committed by row id | Claude | Every number reproducible; nothing fit on test | Re-sampling per run |
| 6 | 2026-09-24 | Report calibration **before and after** temperature scaling, temperatures fit on the calibration split only | Claude | Laya/Jev's pitch is calibrated confidence; temperature scaling is the free baseline fix anyone can apply | Raw ECE only |
| 7 | 2026-09-24 | Project lives outside iCloud, in `~/Downloads` | Moinuddin | iCloud duplicated files in the portfolio repo | `~/Developer` |
| 8 | 2026-09-24 | NLI must score all options in **one batched pass**, like Laya, before any speed comparison | Moinuddin ("do it the way Laya does") | The first NLI runner was unbatched, which flattered Laya | Compare as first measured |
| 9 | 2026-09-24 | Private GitHub repo; SetFit grid on the M4 Pro, everything else on the M1 | Moinuddin | M4 is faster; split avoids duplicate work | One machine only |
| 10 | 2026-09-24 | Add Laya's **own** typed-decisions benchmark (it has a train split) | Moinuddin | "They show DAIR Emotion and typed-decisions along with what we did" — test on their home turf; their docs say all capability there comes from fine-tuning | Only third-party datasets |
| 11 | 2026-09-24 | Apply Laya's published limitations: SMS spam flagged as possibly in its training mix; Banking77 reported as an architectural ceiling (77 options share a 192-token budget); use per-option probabilities, not its `confidence` field (1 − normalised entropy) | Claude | Their own repo documents these; ignoring them would misstate results | Treat all tasks as clean zero-shot |
| 12 | 2026-09-24 | Add an option-order robustness check | Claude | Both Laya's repo and nibzard report answer flips when options are reordered | — |
| 13 | 2026-09-24 | Lead with **accuracy vs speed** | Moinuddin ("the thing about Laya and Jev is speed first") | Their product claim is speed; judge them on it | Accuracy-only tables |
| 14 | 2026-09-24 | Time **every** method on the same machine, including embeddings + LR and SetFit | Claude | Methods that read only the text may be faster than Laya; mixing M1 and M4 timings would be invalid | Latency for Laya and NLI only |
| 15 | 2026-09-24 | Laya's real niche is **changing categories** (zero-shot) — and NLI shares it, so vs NLI the edge is speed, vs SetFit it's flexibility | Moinuddin (niche) · Claude (refinement) | "Laya holds when categories change constantly" | "Laya is just hype" |
| 16 | 2026-09-24 | Include **Jev** directly via its API | Moinuddin (has access) · Claude (recommended) | Laya's Jev numbers are borrowed, not run: "indicative, not a controlled head-to-head". Calibration needs raw per-example outputs | Rely on published Jev numbers |
| 17 | 2026-09-24 | Keep SetFit, and the M4 runs the **full** grid (5 tasks × k∈{8,16,64} × 5 seeds) | Moinuddin | Approved on the M4 with auto permissions | Trimmed grid (3 seeds, k=64 only where undecided) |
| 18 | 2026-09-25 | Time SetFit latency once per (task, k), seed 0, 50 examples | Claude (M4 session proposed it) | Timing loop, not training, was ~10 of ~11 min per run; latency doesn't depend on the training draw | Time every seed |
| 19 | 2026-09-25 | Same-**framework** speed table: Laya also timed on PyTorch MPS; MLX shown separately as "optimised runtime" | Moinuddin (raised MLX) · Claude (fairness fix) | Laya runs on MLX, baselines on PyTorch; part of its lead could be the framework | MLX vs PyTorch as measured |
| 20 | 2026-09-25 | Paired bootstrap 95% CIs on every headline difference | Claude | "X beats Laya" needs error bars; predictions are cached, so it's cheap | Point estimates only |
| 21 | 2026-09-25 | Ship as a **repo** first; a paper (arXiv / Zenodo / workshop) later | Moinuddin | No arXiv endorser lined up yet; the repo is the deliverable either way | Paper now |
| 22 | 2026-09-25 | Defer the RLCD vs cross-entropy ablation | Moinuddin (repo first) | It's the paper's scientific core, not needed for the repo | Run it now |

## Open questions (for Moinuddin)
- Make the repo public once results are final?
- Add the "new categories" experiment to test the changing-categories niche directly?
- Paper later: which venue, and who endorses on arXiv?
