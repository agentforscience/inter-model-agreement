# Resources used

The resource-finding stage was skipped for this run: no `literature_review.md`, `papers/` or `code/` were
present. All resources were gathered during the experiment session.

## Datasets (Hugging Face Hub; see datasets/README.md)
- `mandarjoshi/trivia_qa` (rc.nocontext, validation)
- `cais/mmlu` (test)
- `openai/gsm8k` (main, test)
- `HuggingFaceH4/MATH-500`
- `hendrycks/ethics` (commonsense/test_hard.csv, justice/test_hard.csv)

## Models (4-bit checkpoints, run locally with vLLM 0.31.0)
- Qwen/Qwen2.5-7B-Instruct-AWQ
- Qwen/Qwen3-8B-AWQ
- hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
- hugging-quants/gemma-2-9b-it-AWQ-INT4
- solidrust/Mistral-7B-Instruct-v0.3-AWQ
- tiiuae/Falcon3-7B-Instruct-AWQ
- kaitchup/Olmo-3-7B-Instruct-gptq-w4a16-g128
- kaitchup/Phi-3-mini-4k-instruct-gptq-4bit

## Process notes
- No OpenRouter, OpenAI or Anthropic keys were available, so local open models were used.
- Shared GPUs had only 6–10 GB free each, so 4-bit checkpoints were used for every model.
- Two cached gated checkpoints (google/gemma-7b-it, meta-llama/Llama-3.1-8B-Instruct) lacked config and
  tokenizer files and were replaced by the checkpoints above.
- The answer-equivalence rule for TriviaQA was tightened after error analysis (§5.5 of REPORT.md). Substring
  containment of one-token answers made the agreement relation non-transitive.
