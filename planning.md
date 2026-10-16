# Research Plan: Inter-Model Agreement as a Proxy for Answer Correctness

## Motivation & Novelty Assessment

### Why This Research Matters
Deploying LLMs without labeled data requires cheap signals of when an answer is right. If independent
models agree, practitioners often treat the consensus as trustworthy (LLM juries, ensembles, "wisdom of
the silicon crowd", synthetic-label pipelines). Whether that trust is warranted — and whether it fails in
domains where models share training-induced biases (ethics/moral judgement) — directly affects how
ensembles are used for evaluation, labeling and safety.

### Gap in Existing Work
(No literature review was pre-gathered for this run — `literature_review.md`/`papers/` were absent; this
summary is from background knowledge.) Self-consistency (Wang et al., 2022) shows *intra*-model agreement
predicts correctness for math; ensembles/"LLM-Blender"/"More Agents Is All You Need" (Li et al., 2024)
show voting gains; "Correlated errors in LLMs" (Kim et al., 2025) and "Great models think alike"
(Goel et al., 2025) report that errors across models are correlated. What is missing is a single,
controlled comparison that measures, on the *same* model pool, (a) the agreement–correctness correlation,
(b) majority-vote gain over the best single model, and (c) unsupervised AUROC of agreement — **contrasted
across factual, math and ethical-judgement tasks** where the "shared bias" mechanism should differ.

### Our Novel Contribution
A cross-domain test of a three-part quantitative hypothesis (r ranges, vote gain 3–7 %, AUROC 0.70–0.80)
with 8 open-weight instruction models from 6 developer families, plus error-correlation analysis that
explains *why* agreement fails where it fails.

### Experiment Justification
- Exp 1 (generation): every model answers the same 6 × 500 items greedily — needed to have aligned answers.
- Exp 2 (agreement ↔ correctness correlation, per task): tests sub-hypothesis H1.
- Exp 3 (majority vote vs best model, N=5 subsets and N=8): tests H2.
- Exp 4 (AUROC of agreement, compared to each model's own token-probability confidence): tests H3 and
  provides a stronger baseline than "no signal".
- Exp 5 (pairwise error correlation, "confident-consensus-but-wrong" rate, label-noise inspection): tests the
  proposed *mechanism* (shared systematic errors) and alternative explanations (e.g. noisy ethics labels).

## Research Question
Does the fraction of independent LLMs that agree on an answer predict whether that answer is correct, and
does this differ between factual/math tasks and ethical-judgement tasks?

## Hypothesis Decomposition
- H1: point-biserial r(agreement, correctness) ∈ [0.4, 0.6] for factual & math; r < 0.2 for ethics.
- H2: majority vote of ≥5 models beats the best individual model by 3–7 pp on factual QA; underperforms it on ethics.
- H3: agreement rate alone gives AUROC 0.70–0.80 for predicting correctness on factual tasks.

## Proposed Methodology
### Setup (deviation: forced by environment)
No API keys (OPENROUTER etc.) are available on this host, so real *open-weight* models are run locally with
vLLM on one RTX A6000. Models (≈4–8B, instruction-tuned, 2024–2025): Qwen2.5-7B-Instruct, Qwen3-8B
(non-thinking), Llama-3.1-8B-Instruct, Phi-3-mini-4k-instruct, Gemma-7B-it, OLMo-3-7B-Instruct,
Granite-3.3-8B-Instruct, Falcon3-7B-Instruct.

### Datasets (500 items each, seed 42)
| domain | dataset | format |
|---|---|---|
| factual | TriviaQA rc.nocontext validation | open short answer |
| factual | MMLU test (moral_* subjects excluded) | 4-way MC |
| math | GSM8K test | CoT → number |
| math | MATH-500 | CoT → \boxed{} |
| ethics | ETHICS commonsense (test_hard, short) | binary wrong / not wrong |
| ethics | ETHICS justice (test_hard) | binary reasonable / unreasonable |

### Experimental Steps
1. Build prompts, run each model greedily (T=0), save raw outputs + first-answer-token probability.
2. Parse answers; task-specific equivalence (letter, numeric, normalized LaTeX, normalized string/containment).
3. Plurality answer & agreement rate per item; correctness vs gold.
4. Metrics + bootstrap CIs (1000 resamples), McNemar for vote vs best model, Holm correction.

### Baselines
Best single model (by test accuracy — an *oracle* choice, favourable to the baseline); random single model;
each model's own token probability as a confidence score (supervision-free, single-model).

### Evaluation Metrics
Accuracy; point-biserial r; AUROC; pairwise error correlation (phi on error indicators) and Cohen's κ.

### Statistical Analysis Plan
Bootstrap 95 % CI for r, AUROC, vote − best diff; McNemar exact test (vote vs best); α = 0.05, Holm across
6 datasets. Domain comparisons via bootstrap difference of r.

## Expected Outcomes
Support: factual/math r≈0.4–0.6, AUROC 0.7–0.8, positive vote gain; ethics r<0.2, vote ≤ best.
Refute: ethics agreement as predictive as factual, or factual AUROC outside the range.

## Timeline
Setup 30 min; generation ~1.5 h GPU; analysis 45 min; report 30 min.

## Potential Challenges
Answer-equivalence errors in open answers (mitigate: lenient containment matching; manual spot check);
ethics label noise (inspect high-consensus disagreements with gold); small-model weaknesses (MATH);
format failures (count as "no answer", reported).

## Success Criteria
All 8 × 6 × 500 generations parsed, all three sub-hypotheses evaluated with CIs, and the report states
clearly which parts are supported.

## Deviations from plan (recorded after execution)
- Models: no API keys and only 6–10 GB free per shared GPU, so all 8 models are 4-bit AWQ/GPTQ checkpoints.
  Gemma-7B-it was replaced by Gemma-2-9B-it and GLM-4 by Mistral-7B-v0.3, because cached gated checkpoints
  were incomplete.
- Added a binary factual control (MMLU true/false, `src/data_prep_control.py`) after seeing the main
  results, to separate answer format (binary) from domain (ethics).
- Tightened TriviaQA answer equivalence after error analysis: one-token containment made agreement
  non-transitive.
- Added a half-split "selected best" baseline, an ensemble-size sweep, a one-model-per-family check and
  selective-prediction curves.
