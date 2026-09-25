# System One models, re-examined

![Accuracy on five tasks: Laya, Jev, a 2021 zero-shot NLI model, and SetFit with 16 labels per class](figures/hero_bars.png)

**Laya and Jev are pitched as a new kind of model. Each wins one task. Older, smaller methods win the rest, and run faster.**

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and its open alternative
[Laya](https://github.com/NandhaKishorM/laya) answer typed questions about text with probabilities, in
milliseconds, without generating text. They compare themselves with LLMs. This repo compares them with the
classifiers we already had.

## Claims vs results

| Claim | Result |
|---|---|
| Accurate with no training | Laya wins news topics (0.92), Jev wins ratings (0.57). A 2021 zero-shot NLI model beats Laya on 4 of 5 tasks. |
| Fast | Yes vs LLMs. No vs small classifiers: SetFit is 3–7× faster than Laya. |
| Calibrated | Over-confident on harder tasks. Jev says "100% sure" and is right 77–79% of those times on emotion and ratings. |
| No labels needed | True, but SetFit with 16 labels per class beats Laya on 4 of 5 tasks. |

Credit where due: millisecond decisions without generation are a real win over LLMs.

## Accuracy

1,000 test examples per task. *k* = labels per class, mean of 5 draws.

| Task | Laya | Jev | NLI-base | SetFit k=16 | SetFit k=64 |
|---|---|---|---|---|---|
| AG News (4 topics) | **0.919** | 0.878 | 0.889 | 0.850 | 0.877 |
| Emotion (6) | 0.587 | 0.592 | 0.741 | 0.612 | **0.772** |
| Banking77 (77 intents) | 0.358 | 0.788 | 0.696 | 0.842 | **0.889** |
| SST-5 (1–5 rating) | 0.303 | **0.574** | 0.476 | 0.456 | 0.516 |
| SMS spam¹ | 0.923 | 0.973 | **0.986** | 0.955 | 0.972 |

¹ Spam was in Laya's training data. Confidence intervals for every comparison: [`RESULTS.md`](results/RESULTS.md).

![Accuracy vs labels per class](figures/accuracy_vs_labels.png)

SetFit catches Laya with 8 labels on Banking77, SST-5 and spam, 16 on Emotion, and never on AG News.

## Speed

Median ms per decision on an M4 Pro, one method per process.

| Method | AG News (4 options) | Banking77 (77 options) |
|---|---|---|
| Laya (PyTorch) | 28.4 | 64.2 |
| Laya (MLX) | 20.3 | 50.2 |
| NLI-base | 23.9 | 119.6 |
| SetFit | 8.7 | 9.0 |
| Embeddings + LR | 6.2 | 5.8 |
| Jev (API, with network) | ~830 | ~850 |

![Accuracy vs speed](figures/accuracy_vs_speed.png)

Laya and NLI slow down as options grow. SetFit doesn't.

## Calibration

![Reliability diagrams](figures/reliability.png)

Below the diagonal = over-confident. Both System One models are, on Emotion and SST-5.

## Laya's own benchmark

On [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) we reproduce Laya exactly.

| Method | Accuracy |
|---|---|
| Laya, fine-tuned on this benchmark | **0.767** |
| Jev, zero-shot | 0.736 |
| NLI-base, zero-shot | 0.486 |
| Majority baseline | 0.484 |
| Laya, zero-shot | 0.361 |

Zero-shot, Laya is below the majority baseline. The 0.767 needs training on this benchmark's own data.

## Also found

- Reordering the options changes Laya's answer on 1–6% of examples. NLI: never.
- With 77 options, Laya cuts labels to 3 tokens; some become identical.
- Laya scores 1–7 points under its model card. Its prompts aren't published.

## Which to use

| Situation | Use |
|---|---|
| ~16 labels per category | **SetFit** |
| No labels, changing categories | **NLI zero-shot** (DeBERTa-v3-base) |
| Ratings, no labels, API latency OK | **Jev** |
| Topic routing, no labels, on a Mac | **Laya** |

## Method

- **Same everything:** 1,000 fixed test examples, same wording for every zero-shot model, same machine for speed.
- **Calibration** fixed with temperature scaling on a separate 500-example split, never on test.
- **Every prediction** saved in [`results/preds/`](results/preds/).
- **Bugs we caught:** SetFit trained 1,000 fixed steps instead of one epoch; the first speed test leaked GPU memory into swap. Both fixed and re-run.
- **Limits:** Jev is an API and can change (`jev-1.13.0`, Sept 2026). Five English tasks plus one synthetic benchmark.

## Reproduce

```bash
uv venv --python 3.11 && uv pip install -e ".[dev,laya,nli,fewshot]"
s1x run laya ag_news                     # zero-shot
s1x fewshot setfit ag_news --k 8 16 64   # few-label
python -m s1x.latency --out results/latency_mine.json
s1x report && python scripts/make_figures.py
```

Jev needs `TYPESAFE_API_KEY` in `.env.local`. Laya-MLX needs Apple Silicon.

## Related

[Laya](https://github.com/NandhaKishorM/laya) · [laya-mlx](https://github.com/mizorewww/laya-mlx) ·
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) ·
[decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) (Jev vs LLMs) ·
Yin et al. 2019 (NLI zero-shot) · Tunstall et al. 2022 (SetFit) · Guo et al. 2017 (calibration)

---

By [Moinuddin Shaik](https://moinuddin.app) · full results in [`RESULTS.md`](results/RESULTS.md)
