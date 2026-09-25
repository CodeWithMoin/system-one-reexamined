# System One models, re-examined

![Accuracy on five tasks for Laya, Jev, a 2021 zero-shot NLI model, and SetFit with 16 labels per class](figures/hero_bars.png)

**Laya and Jev are pitched as a new kind of model. On five tasks, each wins exactly one. Older, smaller
methods win the rest — and run several times faster.**

---

## Why this exists

In September 2026, [TypeSafe launched Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a
"System One" model: give it text and a typed question, and it returns calibrated probabilities over your
options in milliseconds, without generating text. [Laya](https://github.com/NandhaKishorM/laya) is the open
alternative.

Jev is benchmarked against LLMs. Laya is benchmarked against Jev. Neither is compared with the classifiers we
already had, even though Laya's architecture — a fine-tuned BERT-style encoder that scores each option and
takes a softmax — is the same idea as NLI zero-shot classifiers from 2019 and 2020.

So this repo asks the question the launches skipped: **is this a new capability, or fast packaging of an
older one?**

## The short answer

Fast packaging of an older idea. Jev improved on some task types; Laya did not. Neither beats a 2021
zero-shot model or a classifier trained on 16 labels per class across the board.

| | What they claim | What we measured |
|---|---|---|
| **Accurate with no training** | Zero-shot typed decisions | Each wins one task: Laya news topics (0.919), Jev ratings (0.574). A 2021 zero-shot NLI model smaller than Laya beats it on 4 of 5 tasks. |
| **Fast** | Milliseconds, 40–200× faster than LLMs | True against LLMs. Not against small classifiers: SetFit is 3–7× faster than Laya on the same machine. |
| **Calibrated** | Confidence you can act on | Both over-confident on the harder tasks. Jev says "100% sure" on up to 62% of answers, and on emotion and ratings it's right 77–79% of those times. |
| **No labels needed** | Skip the labelling | True — but with 16 labels per class, SetFit beats Laya on 4 of 5 tasks. |

What's real deserves the credit: typed decisions in tens of milliseconds, with no text generation, is a
genuine engineering win over LLMs. And Laya's repository is unusually candid about its own limits.

---

## Accuracy

1,000 fixed test examples per task. Zero-shot models get no labels. SetFit and Emb+LR get *k* labelled
examples per class, averaged over 5 random draws.

| Task | Laya | Jev | NLI-base (2021) | SetFit k=16 | SetFit k=64 | Emb+LR k=64 |
|---|---|---|---|---|---|---|
| AG News (4 topics) | **0.919** | 0.878 | 0.889 | 0.850 | 0.877 | 0.871 |
| Emotion (6) | 0.587 | 0.592 | 0.741 | 0.612 | **0.772** | 0.559 |
| Banking77 (77 intents) | 0.358 | 0.788 | 0.696 | 0.842 | 0.889 | **0.895** |
| SST-5 (1–5 rating) | 0.303 | **0.574** | 0.476 | 0.456 | 0.516 | 0.432 |
| SMS spam (yes/no)¹ | 0.923 | 0.973 | **0.986** | 0.955 | 0.972 | 0.951 |

¹ Spam was in Laya's training mix, so it isn't a clean zero-shot test for Laya.
Every comparison with Laya has a paired bootstrap 95% confidence interval in [`results/RESULTS.md`](results/RESULTS.md).

### How many labels does it take to catch them?

![Accuracy against the number of labels per class, with zero-shot models as horizontal lines](figures/accuracy_vs_labels.png)

SetFit passes Laya with **8 labels per class** on Banking77, SST-5 and spam, and with **16** on Emotion. It
never catches Laya on AG News. Against Jev it wins Emotion and Banking77, ties AG News and spam, and trails on
SST-5.

---

## Speed

Median milliseconds per decision, one decision at a time, model load excluded. Measured on an Apple M4 Pro
(24 GB) with each method in its own process and swap flat throughout
([`latency_m4.json`](results/latency_m4.json), with the memory log alongside).

| Method | AG News (4 options) | Banking77 (77 options) |
|---|---|---|
| Laya on PyTorch, the same framework as the rest | 28.4 | 64.2 |
| Laya on MLX, its optimised runtime | 20.3 | 50.2 |
| NLI-large (DeBERTa-v3-large) | 63.8 | 372.9 |
| NLI-base (DeBERTa-v3-base) | 23.9 | 119.6 |
| SetFit | 8.7 | 9.0 |
| Embeddings + logistic regression | 6.2 | 5.8 |
| Jev (API round trip, including the network) | ~830 | ~850 |

![Accuracy against latency on AG News and Banking77](figures/accuracy_vs_speed.png)

Up and to the left is better. Laya and NLI read every option, so they slow down as options grow. SetFit and
logistic regression read only the text, so 77 options cost them nothing extra.

---

## Calibration

![Reliability diagrams for Laya, Jev and NLI-base](figures/reliability.png)

A well-calibrated model sits on the diagonal: when it says 70%, it's right 70% of the time. On Emotion and
SST-5, both System One models sit well below it, claiming more certainty than they deliver. On AG News, all
three are close to honest. After standard temperature scaling (Guo et al., 2017), which works on any
classifier, every model lands in the same range.

---

## On Laya's own benchmark

Laya's headline number comes from [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions):
400 test cases and 2,000 decisions across agent traces, customer service, invoices and security incidents. We
reproduce Laya's published numbers exactly.

| Method | Labels used | Accuracy | Published |
|---|---|---|---|
| Laya, base (zero-shot) | 0 | 0.361 | 0.361 |
| Laya, fine-tuned on this benchmark's train split | 6,000 decisions | **0.767** | 0.766 |
| Jev (zero-shot) | 0 | 0.736 | 0.727 |
| NLI-base (zero-shot) | 0 | 0.486 | – |
| Embeddings + logistic regression, one head per question | 6,000 decisions | 0.564 | – |
| Majority baseline² | train frequencies | 0.484 | 0.461 |

Zero-shot, base Laya scores below the majority baseline, as its own documentation says. The 0.767 comes from
a checkpoint trained on this benchmark's own training data.

² Laya's evaluation script hard-codes 0.461 with no code behind it. Our definition (the most frequent answer per
workflow and question) is documented in `RESULTS.md`; the gap changes no conclusion.

---

## Also worth knowing

- **Laya changes its answer when the options are reordered** — on 1.3% of AG News and 5.7% of Emotion
  examples, across 3 fixed orderings. The NLI model never did.
- **Laya can't read 77 options.** All options share a 192-token budget, so 30 of Banking77's 77 labels are cut
  to 3 tokens, 7 become identical, and the question itself is truncated.
- **Laya scores 1–7 points below its model card** on every task we share. Its original PyTorch release gives the
  same answers as the MLX port on all 200 checked examples, so the gap is likely prompt wording and test sample.
  Laya doesn't publish its prompts.

## Which one should you use?

| Your situation | Use |
|---|---|
| You can label about 16 examples per category | **SetFit** — the most accurate on most tasks, and the fastest |
| No labels, or categories that keep changing | **NLI zero-shot** (DeBERTa-v3-base) — free, local and strong |
| Ratings or scales, no labels, and API latency is fine | **Jev** |
| Topic routing, no labels, on a Mac, where every millisecond counts | **Laya** |

---

## How it was measured

**Contenders.** A majority baseline; Laya zero-shot (the MLX port, plus the original PyTorch release for
timing); Jev through its API; NLI zero-shot with DeBERTa-v3-large and -base, scoring all options for an input
in one batched pass; bi-encoder zero-shot; and two few-label methods — SetFit (paraphrase-mpnet-base-v2 with
the paper's default recipe) and bge-small embeddings with logistic regression — at 8, 16 and 64 labels per
class, 5 random draws each.

**Tasks.** AG News and DAIR Emotion (both in Laya's own results table), Banking77, SST-5, SMS spam, and Laya's
typed-decisions benchmark.

**Fairness.** Every method sees the same 1,000 test examples, fixed in advance and committed by row id. Every
zero-shot method gets the same question and option wording. Temperatures are fit on a separate 500-example
calibration split, never on test. Every prediction is saved in [`results/preds/`](results/preds/), so any number
can be recomputed.

**Metrics.** Accuracy and macro-F1; calibration error (ECE, 15 bins) before and after temperature scaling;
accuracy on the most-confident 80%; MAE and QWK for SST-5; and paired bootstrap 95% confidence intervals.

### Mistakes we caught along the way

- **SetFit was accidentally trained for a fixed 1,000 steps** instead of one epoch — a `max_steps` setting quietly
  overrides `num_epochs`. Profiling caught it, and every affected run was redone with the paper's recipe.
- **The first speed benchmark leaked GPU memory.** Nine models loaded into one process never released their Apple
  GPU memory, and the machine ended up swapping. Each method now runs in its own process; the numbers above come
  from that clean run.

## Limitations

- Jev is an API and can change. These numbers are for `jev-1.13.0` as served in September 2026.
- Five English classification tasks plus one synthetic benchmark. Results may differ for long documents, other
  languages, or workloads that ask many questions about one text, where Laya's question batching helps.
- Laya's missing prompts mean our wording may not be the one it was tuned for.

## Reproduce

```bash
uv venv --python 3.11 && uv pip install -e ".[dev,laya,nli,fewshot]"
s1x splits ag_news emotion banking77 sst5 sms_spam
s1x run laya ag_news                       # any zero-shot method on any task
s1x fewshot setfit ag_news --k 8 16 64     # the few-label grid
s1x td laya                                # Laya's typed-decisions benchmark
python -m s1x.latency --out results/latency_mine.json
s1x report                                 # rebuilds results/RESULTS.md
python scripts/make_figures.py
```

Jev needs `TYPESAFE_API_KEY` in a gitignored `.env.local`. The MLX port of Laya needs Apple Silicon.

## Related work

- [Laya](https://github.com/NandhaKishorM/laya), its [MLX port](https://github.com/mizorewww/laya-mlx), and
  [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
- [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) compares Jev with eight
  LLMs and simple baselines, but not with trained classifiers.
- Yin, Hay and Roth (2019), zero-shot classification as NLI · Tunstall et al. (2022), SetFit · Guo et al. (2017),
  temperature scaling.

---

By [Moinuddin Shaik](https://moinuddin.app). Full tables, per-task notes and latency detail are in
[`results/RESULTS.md`](results/RESULTS.md).
