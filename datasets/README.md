# Datasets

`items.jsonl` (3000 items) is built by `python src/data_prep.py` (seed 42) and `items_control.jsonl` (500 items)
by `python src/data_prep_control.py` (seed 43). Both download from the Hugging Face Hub:

| dataset | HF source | split | n |
|---|---|---|---|
| triviaqa | `mandarjoshi/trivia_qa` (rc.nocontext) | validation | 500 (491 unique ids) |
| mmlu | `cais/mmlu` (all, moral_* excluded) | test | 500 |
| gsm8k | `openai/gsm8k` (main) | test | 500 |
| math500 | `HuggingFaceH4/MATH-500` | test | 500 |
| ethics_cs | `hendrycks/ethics` `data/commonsense/test_hard.csv`, short scenarios | test_hard | 500 |
| ethics_justice | `hendrycks/ethics` `data/justice/test_hard.csv` | test_hard | 500 |
| mmlu_tf (control) | `cais/mmlu` test items not used above, true/false verification | test | 500 |

Each line: `{id, dataset, domain, prompt, gold, meta}` — the exact prompt text sent to every model.
