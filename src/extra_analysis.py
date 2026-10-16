"""Supplementary analyses (run after analyze.py).

1. Bootstrap tests of AUROC / r differences between task groups (incl. the binary factual control).
2. Robustness: one model per developer family (drop qwen3-8b, which shares a family with qwen2.5-7b).
3. Selective prediction: accuracy/coverage of the plurality answer when abstaining below an agreement threshold.
Writes results/extra.json, results/tables/extra_*.md, figures/selective_prediction.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import analyze as A  # noqa: E402
from generate import MODELS  # noqa: E402

rng = np.random.default_rng(7)
B = 2000


def item_table(ds):
    return pd.read_csv(ROOT / f"results/tables/items_{ds}.csv")


def boot_diff(ta, tb, stat):
    """Two-sample bootstrap (independent item sets) of stat(ta) - stat(tb)."""
    d = []
    for _ in range(B):
        a = ta.iloc[rng.integers(0, len(ta), len(ta))]
        b = tb.iloc[rng.integers(0, len(tb), len(tb))]
        d.append(stat(a) - stat(b))
    d = np.array(d)
    return {"diff": float(stat(ta) - stat(tb)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}


auc = lambda t: A.safe_auc(t.vote_correct.values, t.agreement.values)  # noqa: E731
r = lambda t: A.pb_r(t.vote_correct.values, t.agreement.values)  # noqa: E731


def main():
    out = {}
    T = {d: item_table(d) for d in A.DATASETS}
    groups = {"factual(open+MC)": pd.concat([T["triviaqa"], T["mmlu"]]),
              "math": pd.concat([T["gsm8k"], T["math500"]]),
              "ethics": pd.concat([T["ethics_cs"], T["ethics_justice"]]),
              "binary factual control (mmlu_tf)": T["mmlu_tf"],
              "ethics_cs": T["ethics_cs"], "ethics_justice": T["ethics_justice"]}
    comps = [("factual(open+MC)", "ethics"), ("math", "ethics"), ("binary factual control (mmlu_tf)", "ethics"),
             ("factual(open+MC)", "binary factual control (mmlu_tf)"),
             ("binary factual control (mmlu_tf)", "ethics_cs"), ("binary factual control (mmlu_tf)", "ethics_justice")]
    rows = []
    for a, b in comps:
        da, dr = boot_diff(groups[a], groups[b], auc), boot_diff(groups[a], groups[b], r)
        rows.append({"comparison": f"{a} − {b}",
                     "ΔAUROC [95% CI]": f"{da['diff']:+.3f} [{da['ci'][0]:+.3f}, {da['ci'][1]:+.3f}]", "p (AUROC)": round(da["p_two_sided"], 4),
                     "Δr [95% CI]": f"{dr['diff']:+.3f} [{dr['ci'][0]:+.3f}, {dr['ci'][1]:+.3f}]", "p (r)": round(dr["p_two_sided"], 4)})
        out[f"{a} - {b}"] = {"auc": da, "r": dr}
    t = pd.DataFrame(rows)
    open(ROOT / "results/tables/extra_group_diffs.md", "w").write(t.to_markdown(index=False))
    print(t.to_markdown(index=False))

    # ---- 2. one model per family
    models = [m for m in MODELS if m != "qwen3-8b"]
    items, df = A.load("", models)
    rows = []
    for ds in A.DATASETS:
        Aw, C, _ = A.per_item(df, models, ds)
        A_rows = [(iid, [Aw.loc[iid, m] if isinstance(Aw.loc[iid, m], str) else None for m in models]) for iid in Aw.index]
        vc, ag = A.vote_stats(ds, A_rows, C, items, list(range(len(models))))
        acc = C.values.mean(0)
        rows.append({"dataset": ds, "best": f"{models[int(acc.argmax())]} ({acc.max():.3f})", "vote acc": round(vc.mean(), 3),
                     "vote − best (pp)": round(100 * (vc.mean() - acc.max()), 1),
                     "r": round(A.pb_r(vc, ag), 3), "AUROC": round(A.safe_auc(vc, ag), 3)})
    t = pd.DataFrame(rows)
    open(ROOT / "results/tables/extra_one_per_family.md", "w").write(t.to_markdown(index=False))
    out["one_per_family"] = rows
    print(t.to_markdown(index=False))

    # ---- 3. selective prediction
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = []
    for ds in A.DATASETS:
        t = T[ds]
        ths = sorted(t.agreement.unique())
        cov = [np.mean(t.agreement >= th) for th in ths]
        accs = [t.vote_correct[t.agreement >= th].mean() for th in ths]
        ax.plot(cov, accs, marker="o", ms=3, color=A.colors_for(ds) if hasattr(A, "colors_for") else None, label=ds)
        for th in (0.75, 1.0):
            k = t.agreement >= th - 1e-9
            rows.append({"dataset": ds, "threshold": th, "coverage": round(k.mean(), 3),
                         "accuracy (kept)": round(t.vote_correct[k].mean(), 3),
                         "accuracy (all)": round(t.vote_correct.mean(), 3)})
    ax.set_xlabel("coverage (fraction of items answered)"); ax.set_ylabel("accuracy of plurality answer")
    ax.set_title("Selective prediction by abstaining on low inter-model agreement")
    ax.legend(frameon=False, fontsize=7); fig.tight_layout()
    fig.savefig(ROOT / "figures/selective_prediction.png", dpi=150); plt.close(fig)
    t = pd.DataFrame(rows)
    open(ROOT / "results/tables/extra_selective.md", "w").write(t.to_markdown(index=False))
    out["selective"] = rows
    print(t.to_markdown(index=False))
    json.dump(out, open(ROOT / "results/extra.json", "w"), indent=2)


if __name__ == "__main__":
    main()
