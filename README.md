# Inter-Model Agreement as a Proxy for Answer Correctness

Does agreement among independent LLMs predict that an answer is correct, and does that break down for ethical
judgement? We ran 8 open-weight instruction models (7 families, 4-bit, run locally with vLLM) over 7 tasks
of 500 items each: TriviaQA, MMLU, GSM8K, MATH-500, ETHICS-commonsense, ETHICS-justice, and a binary MMLU
true/false control. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **Agreement is a strong label-free correctness signal for factual and math answers.** AUROC is 0.78–0.85 on
  factual tasks and 0.85–0.91 on math; r is 0.45–0.61, except GSM8K at 0.37. When all 8 models agree on an
  open-ended answer, it is correct 96–100% of the time.
- **It is weaker on ethics:** AUROC 0.63–0.71, r 0.20–0.33 (ethics vs factual ΔAUROC −0.12, p < 0.001). About
  half of that gap is **answer format**: a binary *factual* control gets AUROC 0.735, the same as
  ETHICS-commonsense. ETHICS-justice stays significantly worse than the control.
- **Majority voting did not beat the best single model on factual QA** (−1.8 to −2.4 pp, n.s.), contrary to
  the hypothesised +3–7 pp. It helped only on math (+2.0 to +2.4 pp).
- **On ethics, voting is significantly worse than the best model** (−4.8 pp, Holm p = 0.008; −4.2 pp). The
  deficit grows as models are added, and errors are more correlated than on the binary control (φ 0.45–0.49
  vs 0.30), consistent with shared biases.
- **Leave-one-out agreement from other models beats each model's own token-probability confidence** on every
  direct-answer task (e.g. MMLU 0.85 vs 0.72 AUROC).

Deviation from the task spec: no API keys were available, so real open-weight models were run locally in
4-bit instead of GPT, Claude or Gemini.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync    # deps in pyproject.toml
python src/data_prep.py && python src/data_prep_control.py      # datasets/items*.jsonl
python src/run_all.py                                            # 8 models x 3000 items (waits for free GPU memory)
python src/run_all.py --items items_control.jsonl --tag .control # binary factual control
python src/analyze.py && python src/extra_analysis.py            # results/, figures/
```
This needs a GPU with about 8 GB free per model. Generation took about 1.4 GPU-hours on an RTX A6000 and
analysis takes about 2 minutes on CPU. vLLM must have `VLLM_USE_FLASHINFER_SAMPLER=0` set; `generate.py`
sets it, along with `HF_HUB_OFFLINE=1`.

## Layout
```
src/data_prep.py, data_prep_control.py   build item files (exact prompts)
src/generate.py                          one model -> results/model_outputs/<model>[.control].jsonl (vLLM, greedy)
src/run_all.py                           GPU-memory-aware scheduler with retries
src/parse.py                             answer extraction / equivalence / correctness
src/analyze.py                           agreement, vote, AUROC, bootstrap CIs, McNemar, figures
src/extra_analysis.py                    group-difference tests, one-per-family, selective prediction
results/summary.json, extra.json         all metrics;  results/tables/*.md  report tables
results/answers.csv                      parsed answer + correctness for every item x model
figures/*.png                            plots used in REPORT.md
planning.md                              pre-registered plan
```
