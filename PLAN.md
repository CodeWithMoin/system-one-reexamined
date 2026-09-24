# System One models, re-examined

**Question.** "System One" / typed-decision models (TypeSafe's Jev, Convai's Laya) are pitched as a new
class of model. Architecturally, Laya is a fine-tuned ModernBERT cross-encoder that scores each option at
its own `[MASK]` token and softmaxes across options — the same family as BERT multiple-choice heads (2018)
and NLI zero-shot classifiers (2019–20). Jev compares itself only with LLMs; Laya only with Jev. Neither
compares with the classifiers we already had. **Do they beat them — on accuracy, calibration, labels
needed, and latency — or is this an old idea with a new name?**

Both outcomes are publishable. The write-up reports whatever the numbers say.

## What the vendors already admit (Laya model card, Sep 2026)
- Base Laya zero-shot on its own typed-decisions set: **0.362**, below the **0.461** majority baseline;
  the headline 0.766 is a fine-tuned checkpoint.
- Ships over-confident: ECE **0.466 → 0.081** only after refitting a temperature per question type.
- High-cardinality choice is weak: Banking77 **0.425** (Jev: 0.870). Ordinal `score` weakest: SST-5 **0.372**.
- RLCD reward is a strictly proper scoring rule (log + spherical, RPS for ordinal) via REINFORCE.
  The log score is what cross-entropy already optimises — hence ablation A below.

## Contenders (identical tasks, test sets, and label wording)
| id | Method | Labels used |
|---|---|---|
| `majority` | Most frequent class | train prior |
| `laya` | Laya 421M zero-shot (laya-mlx) | 0 |
| `laya-ml` | Laya multilingual 322M zero-shot | 0 |
| `nli` | DeBERTa-v3-large zero-shot NLI (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) | 0 |
| `embed-lr` | Sentence embeddings + logistic regression | k ∈ {8, 16, 64} / class |
| `setfit` | SetFit (contrastive + head) | k ∈ {8, 16, 64} / class |
| `ft-ce` | ModernBERT-large fine-tuned with cross-entropy | k ∈ {8, 16, 64} / class |
| `jev` | Jev via API, black box (optional, if access) | 0 |
| `llm` | Gemini Flash-Lite zero-shot, as a reference ceiling | 0 |

## Tasks (public, and two overlap Laya's card so our numbers can be checked against theirs)
| Task | Type | Classes | Why |
|---|---|---|---|
| AG News | `choice` | 4 | Laya reports 0.950 — reproduce it |
| DAIR Emotion | `choice` | 6 | Laya reports 0.595 — reproduce it |
| Banking77 | `choice` | 77 | high cardinality, Laya's admitted weakness |
| SST-5 | `score` | 5 ordinal | ordinal, Laya's weakest primitive |
| SMS Spam | `noul` | yes/no | proposition with a clear ground truth |

Fixed stratified test subsets (1,000 per task; all of it if smaller), a separate 500-example
calibration split for temperature fitting, and k-shot train draws over **5 seeds** (mean ± std).
Example IDs for every split are committed, so every number is reproducible.

## Metrics
- Accuracy, macro-F1; for SST-5 also mean absolute error and QWK
- Calibration: ECE (15 bins) **before and after** temperature scaling, Brier score, reliability diagram
- Selective prediction: accuracy at 50/80/95% coverage (is the confidence useful for routing?)
- Latency p50/p95 per decision on the same M1 16 GB, model load excluded; parameter count

## Experiments
1. **Main table** — every contender × every task, at 0 / 8 / 16 / 64 labels.
2. **Labels-to-parity** — how many labels per class SetFit / fine-tuned CE need to match zero-shot Laya.
3. **Calibration** — does RLCD give better confidence than a temperature-scaled NLI or CE model?
4. **Ablation A (stretch)** — same backbone and head, cross-entropy vs RLCD, same data: does the RL buy
   anything over supervised training with the log score?

## Hygiene
- Label names and instructions are identical for every zero-shot method; no prompt tuning on test.
- Nothing is fit on the test split; temperatures come from the calibration split only.
- Laya's context is 512 tokens; inputs longer than that are counted and reported, not silently cut.
- Raw predictions (per example, per method, probabilities) are cached as JSONL and committed.

## Deliverables
1. This repo: harness, cached predictions, figures, `RESULTS.md`.
2. Case study on moinuddin.app (#10), with the main table and reliability diagrams.
3. Lab tab on the site: pick a task and a label budget; the leaderboard reorders from the real runs.
   All precomputed — no server.

## Order of work
1. Harness: task loaders + splits, metrics, prediction cache (this commit)
2. Zero-shot runners: laya, nli, majority → first table on AG News and Emotion (reproduction check)
3. k-shot runners: embed-lr, setfit, ft-ce
4. Remaining tasks, calibration, selective prediction, latency
5. Figures + write-up; site case study + Lab tab
6. Stretch: ablation A; Jev if API access
