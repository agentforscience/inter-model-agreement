"""Run one model (vLLM, greedy) over all items and save raw outputs.

Usage: python src/generate.py <model_key> [--limit N]
Output: results/model_outputs/<model_key>.jsonl with {id, text, first_tok_prob, n_tokens}.
`first_tok_prob` is the probability of the first generated token, a single-model confidence baseline
(meaningful for the direct-answer tasks; for CoT tasks it is not used).
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # gated models (Llama, Gemma) are served from the local cache
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # flashinfer JIT build fails on this host
import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 4-bit (AWQ / GPTQ, W4A16) checkpoints: the shared GPUs only have ~6-8 GB free each, and using the same
# precision for every model keeps the comparison fair. 8 models from 7 developer families.
MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "qwen3-8b": "Qwen/Qwen3-8B-AWQ",  # run in non-thinking mode
    "llama3.1-8b": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "gemma2-9b": "hugging-quants/gemma-2-9b-it-AWQ-INT4",
    "mistral-7b": "solidrust/Mistral-7B-Instruct-v0.3-AWQ",
    "falcon3-7b": "tiiuae/Falcon3-7B-Instruct-AWQ",
    "olmo3-7b": "kaitchup/Olmo-3-7B-Instruct-gptq-w4a16-g128",
    "phi3-mini": "kaitchup/Phi-3-mini-4k-instruct-gptq-4bit",
}
MAX_TOKENS = {"mmlu": 8, "triviaqa": 32, "ethics_cs": 8, "ethics_justice": 8, "mmlu_tf": 8, "gsm8k": 768, "math500": 1536}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--items", default="items.jsonl", help="file in datasets/")
    ap.add_argument("--tag", default="", help="output suffix, e.g. .control")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open(ROOT / "datasets" / args.items)]
    if args.limit:
        by = {}
        for it in items:
            by.setdefault(it["dataset"], []).append(it)
        items = [it for v in by.values() for it in v[: args.limit]]

    name = MODELS[args.model]
    max_len = 2560  # longest prompt (~700 tok) + max_tokens 1536 fits
    # GPUs are shared with other jobs: claim (free memory - 1.5 GB) as a fraction of the device total.
    import torch
    free, total = torch.cuda.mem_get_info()
    util = float(os.environ.get("GPU_UTIL", min(12.5 * 2**30 / total, (free - 1.5 * 2**30) / total)))
    print(f"GPU free {free/2**30:.1f} GB -> gpu_memory_utilization={util:.2f}")
    quant = os.environ.get("QUANT") or None  # None -> use the checkpoint's own AWQ/GPTQ config
    llm = LLM(model=name, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=util,
              seed=42, trust_remote_code=False, quantization=quant,
              max_num_seqs=64, max_num_batched_tokens=4096,
              enforce_eager=free < 12 * 2**30)  # skip CUDA-graph memory when tight
    tok = llm.get_tokenizer()

    def render(prompt):
        msgs = [{"role": "user", "content": prompt}]
        kw = {"enable_thinking": False} if "Qwen3" in name else {}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    prompts = [render(it["prompt"]) for it in items]
    params = [SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS[it["dataset"]], logprobs=1, seed=42)
              for it in items]
    t0 = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t0

    out_dir = ROOT / "results/model_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.tag + (f".limit{args.limit}" if args.limit else "")
    with open(out_dir / f"{args.model}{suffix}.jsonl", "w") as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            p0 = None
            if c.logprobs:
                lp = c.logprobs[0][c.token_ids[0]].logprob
                p0 = math.exp(lp)
            f.write(json.dumps({"id": it["id"], "text": c.text, "first_tok_prob": p0, "cum_logprob": c.cumulative_logprob,
                                "n_tokens": len(c.token_ids), "finish": c.finish_reason}) + "\n")
    meta = {"model": name, "n_items": len(items), "seconds": dt, "max_model_len": max_len,
            "temperature": 0.0, "max_tokens": MAX_TOKENS, "quantization": quant}
    json.dump(meta, open(out_dir / f"{args.model}{suffix}.meta.json", "w"), indent=2)
    print(f"done {args.model}: {len(items)} items in {dt:.0f}s")


if __name__ == "__main__":
    main()
