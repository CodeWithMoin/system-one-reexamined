# How this project came about, and how it was done

## 1. Where it started
While building his portfolio, Moinuddin asked whether to add **Laya** to the site's Lab as a
game — Laya being an open "System One" decision model (the open alternative to TypeSafe's
**Jev**), running on Macs through `laya-mlx`. Two problems surfaced: laya-mlx is someone else's
work, and it can't run in a browser or on the site's host. So the question changed from *"show
Laya"* to *"build something of his own on top of it."*

## 2. Finding the real question
A first idea — a cascade that answers with Laya and escalates to an LLM — was set aside when
Moinuddin pushed on what Laya actually is: *a transformer that doesn't write text, it gives a
softmax — and we had this before.* Reading both vendors' material confirmed it:

- **Laya** is ModernBERT-large, fine-tuned, with each option scored at its own `[MASK]` token and
  softmaxed — the same family as BERT multiple-choice heads (2018) and NLI zero-shot classifiers
  (2019–20). Its repo is candid: zero-shot it is below the majority baseline on its own benchmark,
  it ships over-confident, collapses outside English, and caps out on many-option questions.
- **Jev** discloses no weights or architecture; it compares itself only with LLMs. "Can't
  hallucinate" is true by construction (it can only pick a given option), not a claim of accuracy.
- An existing benchmark (`nibzard/decision-model-benchmark`) compares Jev with LLMs and trivial
  baselines — **nobody had compared these models with the trained classifiers we already had.**

That gap became the project: ***System One models, re-examined.***

## 3. The design
- **Contenders:** majority baseline; Laya and Laya-multilingual (zero-shot); NLI zero-shot
  (DeBERTa-v3-large); embeddings + logistic regression and SetFit at 8/16/64 labels per class; Jev
  via API; later a plain fine-tuned ModernBERT.
- **Tasks:** AG News and DAIR Emotion (both in Laya's own table — reproduction checks), Banking77
  (many categories), SST-5 (rating scale), SMS spam (yes/no), and Laya's own typed-decisions
  benchmark (their home turf).
- **Hygiene:** fixed splits committed by row id, identical wording for every zero-shot method,
  temperatures fit on a separate calibration split, every prediction cached to disk.
- **Metrics:** accuracy, macro-F1, calibration error (ECE) before and after temperature scaling,
  accuracy on the most-confident 50/80/95%, and latency on one machine.

## 4. Doing it
1. **Harness** — splits, metrics (with tests), prediction cache, Laya and NLI runners.
   A network glitch turned out to be AdGuard DNS filtering blocking Python's package host.
2. **First results** (AG News, Emotion, 1,000 examples each): Laya reproduced its own Emotion
   number (0.587 vs 0.595); NLI beat Laya by 12 points on Emotion, Laya won AG News by 2; Laya
   was badly over-confident on Emotion (ECE 0.304) and, after temperature scaling, no better
   calibrated than NLI (0.040 vs 0.035); Laya was 7–14× faster than unbatched NLI.
3. **Fairness fixes**, each prompted by a question from Moinuddin: batch NLI like Laya; time every
   method on the same machine; time Laya in the same framework (PyTorch) as the baselines.
4. **Scaling out** — a background agent on the M1 runs the zero-shot tasks, the embeddings
   baseline, typed-decisions and Jev; a Claude session on the M4 Pro (via Remote Control) runs
   the SetFit grid. Both push to one private repo; their files never overlap.
5. **Course corrections** — the SetFit grid's cost was mostly a latency-timing loop, not training,
   so timing now runs once per (task, k); Laya's documented limits were folded into how results
   are reported (training-mix caveat on spam, the 77-option ceiling on Banking77).

## 5. What the project is testing
Four vendor claims, each against evidence:

| Claim | Test |
|---|---|
| Fast | Latency against every method, same machine, same framework |
| Accurate without training | Zero-shot Laya/Jev vs 2020-era NLI zero-shot |
| Calibrated | ECE before and after the fix anyone can apply |
| No labels needed | How many labels SetFit / embeddings + LR need to catch up |

Plus: on Laya's own benchmark, does its fine-tuned model beat cheap classifiers trained on the
same data?

## 6. Where it's going
Results land in `results/RESULTS.md`. Then: Moinuddin reads them and decides the story; it becomes
project #10 on moinuddin.app with a Lab slider built from the real runs; a paper can follow.

See `DECISIONS.md` for every call and who made it, and `PLAN.md` for the full protocol.
