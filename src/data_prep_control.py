"""Binary *factual* control task (format-matched to the binary ethics tasks).

MMLU true/false verification: question + one proposed answer, which is the gold option with p=0.5 and a
uniformly random distractor otherwise; the model answers "true" or "false". Uses MMLU test items that are
NOT in the main MMLU sample (moral_* subjects excluded). Output: datasets/items_control.jsonl.
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 43
N = 500
ROOT = Path(__file__).resolve().parent.parent
PROMPT = ("The following is a question about {subject}, followed by a proposed answer.\n\n"
          "Question: {question}\n\nProposed answer: {cand}\n\n"
          "Is the proposed answer correct? Answer with exactly one word: true or false.")


def main():
    rng = random.Random(SEED)
    used = {json.loads(l)["id"] for l in open(ROOT / "datasets/items.jsonl") if l.startswith('{"id": "mmlu-')}
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    pool = [i for i, s in enumerate(mmlu["subject"]) if not s.startswith("moral") and f"mmlu-{i}" not in used]
    rng.shuffle(pool)
    out = []
    for i in sorted(pool[:N]):
        r = mmlu[i]
        is_true = rng.random() < 0.5
        k = r["answer"] if is_true else rng.choice([j for j in range(4) if j != r["answer"]])
        out.append(dict(id=f"mmlu_tf-{i}", dataset="mmlu_tf", domain="factual",
                        prompt=PROMPT.format(subject=r["subject"].replace("_", " "), question=r["question"],
                                             cand=r["choices"][k]),
                        gold="true" if is_true else "false", meta={"subject": r["subject"]}))
    with open(ROOT / "datasets/items_control.jsonl", "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(len(out), sum(it["gold"] == "true" for it in out), "true")


if __name__ == "__main__":
    main()
