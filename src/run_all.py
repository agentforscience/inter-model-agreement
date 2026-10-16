"""Opportunistic scheduler for src/generate.py on shared GPUs.

GPUs on this host are shared with other jobs whose memory use fluctuates, so for each model we wait for
a GPU with >= MIN_FREE_GB free, launch generate.py there, and retry on failure (e.g. OOM at startup).
Up to MAX_PAR models run concurrently on different GPUs.

Usage: python src/run_all.py [--limit N] [models...]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from generate import MODELS  # noqa: E402

MIN_FREE_GB = 7.0
MIN_FREE_OVERRIDE = {"phi3-mini": 5.0}  # 3.8B model needs much less
MAX_PAR = 4
MAX_TRIES = 40


def free_gb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) / 1024 for a, b in (l.split(",") for l in out.strip().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("models", nargs="*")
    args = ap.parse_args()
    todo = args.models or list(MODELS)
    suffix = args.tag + (f".limit{args.limit}" if args.limit else "")
    todo = [m for m in todo if not (ROOT / f"results/model_outputs/{m}{suffix}.jsonl").exists()]
    tries = {m: 0 for m in todo}
    running = {}  # model -> (proc, gpu, start)
    while todo or running:
        for m, (p, g, t0) in list(running.items()):
            if p.poll() is not None:
                del running[m]
                ok = (ROOT / f"results/model_outputs/{m}{suffix}.jsonl").exists()
                print(f"[{time.strftime('%H:%M:%S')}] {m} on GPU{g} exited rc={p.returncode} ok={ok} "
                      f"after {time.time() - t0:.0f}s", flush=True)
                if not ok:
                    tries[m] += 1
                    if tries[m] < MAX_TRIES:
                        todo.append(m)
        if todo and len(running) < MAX_PAR:
            busy = set()  # jobs cap their own memory, so several may share a GPU if it has room
            fr = free_gb()
            pick = None
            for m in todo:  # first model in queue that fits on some idle GPU
                need = MIN_FREE_OVERRIDE.get(m, MIN_FREE_GB)
                cand = [(f, g) for g, f in fr.items() if g not in busy and f >= need]
                if cand:
                    pick = (m, *max(cand))
                    break
            if pick:
                m, f, g = pick
                todo.remove(m)
                cmd = [sys.executable, str(ROOT / "src/generate.py"), m] + (["--limit", str(args.limit)] if args.limit else []) + ["--items", args.items, "--tag", args.tag]
                log = open(ROOT / f"logs/gen_{m}{suffix}.log", "a")
                env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=str(g))
                running[m] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT), g, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] launch {m} on GPU{g} ({f:.1f} GB free), try {tries[m] + 1}", flush=True)
                time.sleep(60)  # let it claim memory before considering the next launch
                continue
        time.sleep(15)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
