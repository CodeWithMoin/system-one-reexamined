# System One models, re-examined

**Are "System One" decision models — TypeSafe's Jev and its open counterpart Laya — a new capability,
or fast packaging of classifiers we already had?**

Both are pitched as a new class of model: send text and a typed question, get calibrated probabilities
over the options in milliseconds, no text generation. Laya is a fine-tuned ModernBERT cross-encoder that
scores each option at its own `[MASK]` token and softmaxes — the same family as BERT multiple-choice
heads (2018) and NLI zero-shot classifiers (2019–20). Jev compares itself only with LLMs; Laya compares
itself only with Jev. Nobody had compared them with the classifiers we already had. This repo does.

## Answer

**Fast packaging of an older idea, not a new capability.** Jev improved on some task types; Laya did not.
Neither beats a 2021 zero-shot model or a 16-label classifier across the board.

- **Each System One model owns one task type.** Laya is the most accurate model we ran on news topics
  (AG News 0.919 — no amount of labels caught it). Jev is the most accurate on ratings (SST-5 0.574).
- **Everywhere else, older methods match or beat them.** A 2021 zero-shot NLI model smaller than Laya
  (DeBERTa-v3-base) beats Laya on 4 of 5 tasks with no labels. SetFit with 16 labels per class beats Laya
  on 4 of 5.
- **They are fast next to LLMs, not next to small classifiers.** On the same machine and framework, SetFit
  is 3–7× faster than Laya and logistic regression on embeddings 5–11×; their cost doesn't grow with the
  number of options, Laya's does.
- **"Calibrated" isn't borne out.** Both ship over-confident on the harder tasks, and after standard
  temperature scaling (2017) they're no better calibrated than anything else. Jev often reports exactly
  100% confidence and is right 77–79% of those times on Emotion and SST-5.
- **On Laya's own benchmark, its numbers reproduce exactly** (base 0.361, fine-tuned 0.7665) — and base
  Laya zero-shot sits below the majority baseline, as its own docs say. The 0.766 is a checkpoint trained on
  that benchmark's train split.

What's real deserves the credit: decisions in tens of milliseconds with no generation is a genuine
engineering win over LLMs, and Laya's repository is unusually candid about its own limits.

## Accuracy (1,000 fixed test examples per task)

| Task | Laya | Jev | NLI-base (2021) | SetFit k=16 | SetFit k=64 | Emb+LR k=64 |
|---|---|---|---|---|---|---|
| AG News (4 topics) | **0.919** | 0.878 | 0.889 | 0.850 | 0.877 | 0.871 |
| Emotion (6) | 0.587 | 0.592 | 0.741 | 0.612 | **0.772** | 0.559 |
| Banking77 (77 intents) | 0.358 | 0.788 | 0.696 | 0.842 | 0.889 | **0.895** |
| SST-5 (1–5 rating) | 0.303 | **0.574** | 0.476 | 0.456 | 0.516 | 0.432 |
| SMS spam (yes/no)¹ | 0.923 | 0.973 | **0.986** | 0.955 | 0.972 | 0.951 |

Zero-shot: Laya, Jev, NLI-base. k = labelled examples per class, mean of 5 random draws. Paired
bootstrap 95% CIs for every comparison with Laya are in [`results/RESULTS.md`](results/RESULTS.md).
¹ Spam was in Laya's training mix, so it is not a clean zero-shot test for Laya.

![Accuracy vs labels](figures/accuracy_vs_labels.png)

## Speed (median ms per decision, one at a time, model load excluded)

M4 Pro 24 GB, each method in its own process, swap flat throughout
([`results/latency_m4.json`](results/latency_m4.json), memory log alongside).

| Method | AG News (4 options) | Banking77 (77 options) |
|---|---|---|
| Laya, PyTorch (same framework as the rest) | 28.4 | 64.2 |
| Laya, MLX (its optimised runtime) | 20.3 | 50.2 |
| NLI-large (DeBERTa-v3-large) | 63.8 | 372.9 |
| NLI-base (DeBERTa-v3-base) | 23.9 | 119.6 |
| SetFit | 8.7 | 9.0 |
| Embeddings + logistic regression | 6.2 | 5.8 |
| Jev (API round trip, includes network) | ~830 | ~850 |

![Accuracy vs speed](figures/accuracy_vs_speed.png)

![Reliability diagrams](figures/reliability.png)

## Other findings

- **Option order changes Laya's answer** on 1.3% of AG News and 5.7% of Emotion examples (3 fixed
  permutations); NLI's never changes.
- **Laya can't read 77 options.** Options share a 192-token budget: 30 of Banking77's 77 labels are cut to
  3 tokens, 7 collapse into identical strings, and the instruction itself is truncated.
- **Laya scores 1–7 points below its model card** on every shared task. Upstream PyTorch Laya gives the same
  answers as the MLX port (200/200 agreement), so the likely causes are wording and test subsets; its
  prompts aren't published.

## Method

- **Contenders:** majority baseline; Laya (zero-shot, laya-mlx; upstream PyTorch for timing); Jev (API);
  NLI zero-shot with DeBERTa-v3-large and -base (all options for one input scored in one batched pass);
  bi-encoder zero-shot; SetFit (paraphrase-mpnet-base-v2, paper-default recipe: 20 pair-iterations, one epoch)
  and bge-small embeddings + logistic regression, each at k ∈ {8, 16, 64} × 5 seeds.
- **Tasks:** AG News, DAIR Emotion (both in Laya's own table), Banking77, SST-5, SMS spam, and Laya's own
  typed-decisions benchmark.
- **Hygiene:** fixed test (1,000), calibration (500) and k-shot splits committed by row id; identical
  instruction and label wording for every zero-shot method; temperatures fit only on the calibration split;
  every per-example prediction cached in `results/preds/`.
- **Metrics:** accuracy, macro-F1, ECE (15 bins) before and after temperature scaling, accuracy on the
  most-confident 80%, MAE/QWK for SST-5, paired bootstrap CIs.

## Reproduce

```bash
uv venv --python 3.11 && uv pip install -e ".[dev,laya,nli,fewshot]"
s1x splits ag_news emotion banking77 sst5 sms_spam
s1x run laya ag_news                       # any zero-shot method × task
s1x fewshot setfit ag_news --k 8 16 64     # few-label grid
python -m s1x.latency --out results/latency_mine.json
s1x report                                 # rebuilds results/RESULTS.md
python scripts/make_figures.py
```

Jev needs `TYPESAFE_API_KEY` in a gitignored `.env.local`. Laya-MLX needs Apple Silicon.

## Related work

- [Laya](https://github.com/NandhaKishorM/laya) and its [MLX port](https://github.com/mizorewww/laya-mlx);
  [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
- [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) — Jev against
  eight LLMs and trivial baselines (no trained classifiers).
- Yin et al. 2019 (NLI zero-shot classification); Tunstall et al. 2022 (SetFit); Guo et al. 2017
  (temperature scaling).

Full tables, per-task caveats, latency detail and Laya's own benchmark: [`results/RESULTS.md`](results/RESULTS.md).
