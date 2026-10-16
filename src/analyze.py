"""Agreement-vs-correctness analysis over all model outputs.

Reads datasets/items.jsonl and results/model_outputs/<model>.jsonl, writes
  results/answers.csv           (long table: item x model -> answer, correct, confidence)
  results/summary.json          (all metrics with bootstrap CIs)
  results/tables/*.csv / *.md   (report tables)
  figures/*.png
Usage: python src/analyze.py [--suffix .limit5]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from generate import MODELS  # noqa: E402
from parse import correct, equiv, extract  # noqa: E402

DATASETS = ["triviaqa", "mmlu", "gsm8k", "math500", "ethics_cs", "ethics_justice", "mmlu_tf"]
# mmlu_tf = binary factual control (format-matched to the binary ethics tasks); not pooled into "factual".
DOMAIN = {"triviaqa": "factual", "mmlu": "factual", "gsm8k": "math", "math500": "math",
          "ethics_cs": "ethics", "ethics_justice": "ethics", "mmlu_tf": "control"}
DIRECT = {"triviaqa", "mmlu", "ethics_cs", "ethics_justice", "mmlu_tf"}  # single-pass answers -> token prob is a confidence
B = 1000
SEED = 42


# ------------------------------------------------------------------ voting

def plurality(dataset, answers, order=None):
    """Plurality answer among `answers` (list, None = unparseable).

    Each distinct answer's support = number of models whose answer is equivalent to it. Ties are broken
    by the fixed model order (deterministic). Returns (answer, support_count, index_of_representative).
    """
    best, best_n, best_i = None, 0, None
    for i, a in enumerate(answers):
        if a is None:
            continue
        n = sum(equiv(dataset, a, b) for b in answers)
        if n > best_n:
            best, best_n, best_i = a, n, i
    return best, best_n, best_i


def loo_agreement(dataset, answers):
    """For each model i: fraction of the *other* models whose answer is equivalent to model i's answer."""
    n = len(answers)
    out = []
    for i, a in enumerate(answers):
        if a is None:
            out.append(0.0)
        else:
            out.append(sum(equiv(dataset, a, answers[j]) for j in range(n) if j != i) / (n - 1))
    return out


# ------------------------------------------------------------------ stats helpers

def safe_auc(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


def pb_r(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    if y.std() == 0 or s.std() == 0:
        return np.nan
    return float(np.corrcoef(y, s)[0, 1])


def boot_ci(fn, n, rng, b=B):
    """Percentile bootstrap over items; fn(idx) -> statistic."""
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_exact(a, b):
    """a, b boolean arrays (correctness of two systems on same items)."""
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 1.0
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ main

def load(suffix, models):
    items = [json.loads(l) for f in ["items.jsonl", "items_control.jsonl"] if (ROOT / "datasets" / f).exists()
             for l in open(ROOT / "datasets" / f)]
    items = {it["id"]: it for it in items}
    rows = []
    for m in models:
        files = [ROOT / f"results/model_outputs/{m}{suffix}.jsonl"]
        files += [f for f in [ROOT / f"results/model_outputs/{m}.control{suffix}.jsonl"] if f.exists()]
        for l in (line for f in files for line in open(f)):
            d = json.loads(l)
            it = items[d["id"]]
            a = extract(it["dataset"], d["text"])
            rows.append(dict(id=d["id"], dataset=it["dataset"], model=m, answer=a,
                             correct=correct(it["dataset"], a, it["gold"]),
                             tok_prob=d["first_tok_prob"], seq_prob=float(np.exp(d.get("cum_logprob") or 0)),
                             finish=d["finish"], n_tokens=d["n_tokens"]))
    df = pd.DataFrame(rows)
    # TriviaQA rc.nocontext repeats a few question_ids (identical question text); keep one copy of each.
    df = df.drop_duplicates(subset=["id", "model"], keep="first").reset_index(drop=True)
    return items, df


def per_item(df, models, ds):
    """Wide per-item table for one dataset with agreement statistics."""
    sub = df[df.dataset == ds]
    A = sub.pivot(index="id", columns="model", values="answer")[models]
    C = sub.pivot(index="id", columns="model", values="correct")[models].astype(bool)
    P = sub.pivot(index="id", columns="model", values="seq_prob")[models]
    return A, C, P


def vote_stats(ds, A_rows, C, items, model_idx):
    """Majority/plurality vote over a subset of models; returns (vote_correct, agreement_rate)."""
    vc, ag = [], []
    for iid, answers in A_rows:
        ans = [answers[k] for k in model_idx]
        a, n, _ = plurality(ds, ans)
        vc.append(correct(ds, a, items[iid]["gold"]))
        ag.append(n / len(ans))
    return np.array(vc), np.array(ag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    models = [m for m in MODELS if (ROOT / f"results/model_outputs/{m}{args.suffix}.jsonl").exists()]
    print("models:", models)
    items, df = load(args.suffix, models)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    df.to_csv(ROOT / f"results/answers{args.suffix}.csv", index=False)

    summary = {"models": models, "datasets": {}}
    item_tables = {}
    global DATASETS
    DATASETS = [d for d in DATASETS if (df.dataset == d).sum() == len(models) * df[df.dataset == d].id.nunique()
                and (df.dataset == d).any()]
    print("datasets:", DATASETS)
    for ds in DATASETS:
        A, C, P = per_item(df, models, ds)
        ids = list(A.index)
        n = len(ids)
        A_rows = [(iid, [A.loc[iid, m] if isinstance(A.loc[iid, m], str) else None for m in models]) for iid in ids]
        Cm = C.values  # n x M bool
        acc = Cm.mean(0)
        best = int(np.argmax(acc))

        # --- full-ensemble plurality vote
        vote_c, agree = vote_stats(ds, A_rows, C, items, list(range(len(models))))
        # unanimity / supermajority diagnostics
        n_distinct = []
        for _, ans in A_rows:
            reps = []
            for a in ans:
                if a is not None and not any(equiv(ds, a, r) for r in reps):
                    reps.append(a)
            n_distinct.append(len(reps))
        # mean accuracy across models per item ("item easiness") — oracle quantity, for reference only
        easiness = Cm.mean(1)

        # --- H1/H3: agreement vs plurality-answer correctness
        r = pb_r(vote_c, agree)
        auc = safe_auc(vote_c, agree)
        r_ci = boot_ci(lambda idx: pb_r(vote_c[idx], agree[idx]), n, rng)
        auc_ci = boot_ci(lambda idx: safe_auc(vote_c[idx], agree[idx]), n, rng)

        # --- per-model leave-one-out agreement as confidence for that model's own answer
        loo = np.array([loo_agreement(ds, ans) for _, ans in A_rows])  # n x M
        per_model = {}
        for k, m in enumerate(models):
            y = Cm[:, k]
            d = {"acc": float(acc[k]), "loo_r": pb_r(y, loo[:, k]), "loo_auc": safe_auc(y, loo[:, k]),
                 "parse_fail": float(np.mean([ans[k] is None for _, ans in A_rows]))}
            if ds in DIRECT:
                pk = P[m].values.astype(float)
                d["selfprob_auc"] = safe_auc(y, pk)
                d["selfprob_r"] = pb_r(y, pk)
            per_model[m] = d

        # --- H2: vote vs best model (oracle-chosen best) & vs mean model
        diff = vote_c.mean() - acc[best]
        diff_ci = boot_ci(lambda idx: vote_c[idx].mean() - Cm[idx][:, best].mean(), n, rng)
        p_mc = mcnemar_exact(vote_c, Cm[:, best])
        # non-oracle baseline: pick the best model on a random half, evaluate vote & that model on the other half
        split_diffs = []
        for _ in range(200):
            perm = rng.permutation(n)
            sel, ev = perm[: n // 2], perm[n // 2:]
            k_sel = int(np.argmax(Cm[sel].mean(0)))
            split_diffs.append(vote_c[ev].mean() - Cm[ev, k_sel].mean())
        split_diffs = np.array(split_diffs)
        # all 5-model subsets: vote vs best-in-subset
        sub5 = []
        for S in itertools.combinations(range(len(models)), 5):
            vc5, _ = vote_stats(ds, A_rows, C, items, list(S))
            sub5.append(vc5.mean() - acc[list(S)].max())
        sub5 = np.array(sub5)

        # --- ensemble-size sweep: for each k, all (or up to 40 random) k-subsets
        sweep = {}
        allc = list(range(len(models)))
        for k in range(3, len(models) + 1):
            combs = list(itertools.combinations(allc, k))
            if len(combs) > 40:
                combs = [combs[i] for i in rng.choice(len(combs), 40, replace=False)]
            aucs, gains = [], []
            for S in combs:
                vck, agk = vote_stats(ds, A_rows, C, items, list(S))
                aucs.append(safe_auc(vck, agk))
                gains.append(vck.mean() - acc[list(S)].max())
            sweep[k] = {"auc": float(np.nanmean(aucs)), "vote_minus_best": float(np.mean(gains)),
                        "n_subsets": len(combs)}
        # --- error correlation between model pairs
        E = (~Cm).astype(float)
        phis, kappas, agree_pairs = [], [], []
        for i, j in itertools.combinations(range(len(models)), 2):
            if E[:, i].std() > 0 and E[:, j].std() > 0:
                phis.append(np.corrcoef(E[:, i], E[:, j])[0, 1])
            both_wrong = np.mean(E[:, i] * E[:, j])
            # P(same wrong answer | both wrong)
            same = [equiv(ds, A_rows[t][1][i], A_rows[t][1][j]) for t in range(n) if E[t, i] and E[t, j]]
            agree_pairs.append(np.mean(same) if same else np.nan)
        # "confident consensus but wrong": >= 75% agree and plurality wrong
        hi = agree >= 0.75
        summary["datasets"][ds] = {
            "domain": DOMAIN[ds], "n": n,
            "model_acc": {m: float(acc[k]) for k, m in enumerate(models)},
            "best_model": models[best], "best_acc": float(acc[best]), "mean_model_acc": float(acc.mean()),
            "vote_acc": float(vote_c.mean()), "vote_minus_best": float(diff), "vote_minus_best_ci": diff_ci,
            "vote_minus_mean": float(vote_c.mean() - acc.mean()),
            "mcnemar_p": p_mc,
            "split_vote_minus_selected_best": float(split_diffs.mean()),
            "split_vote_minus_selected_best_q": [float(np.percentile(split_diffs, 2.5)), float(np.percentile(split_diffs, 97.5))],
            "sub5_vote_minus_best_mean": float(sub5.mean()), "sub5_frac_positive": float(np.mean(sub5 > 0)),
            "sub5_min": float(sub5.min()), "sub5_max": float(sub5.max()),
            "agree_r": r, "agree_r_ci": r_ci, "agree_auc": auc, "agree_auc_ci": auc_ci,
            "mean_agreement": float(agree.mean()), "frac_unanimous": float(np.mean(agree == 1)),
            "acc_when_unanimous": float(vote_c[agree == 1].mean()) if np.any(agree == 1) else None,
            "frac_hi_agree": float(hi.mean()),
            "acc_when_hi_agree": float(vote_c[hi].mean()) if hi.any() else None,
            "frac_hi_agree_wrong": float(np.mean(hi & ~vote_c)),
            "mean_error_phi": float(np.nanmean(phis)),
            "mean_same_wrong_answer": float(np.nanmean(agree_pairs)),
            "mean_n_distinct": float(np.mean(n_distinct)),
            "per_model": per_model,
            "size_sweep": sweep,
            "loo_auc_mean": float(np.nanmean([v["loo_auc"] for v in per_model.values()])),
            "loo_r_mean": float(np.nanmean([v["loo_r"] for v in per_model.values()])),
        }
        if ds in DIRECT:
            summary["datasets"][ds]["selfprob_auc_mean"] = float(np.nanmean([v["selfprob_auc"] for v in per_model.values()]))
        item_tables[ds] = pd.DataFrame({"id": ids, "agreement": agree, "vote_correct": vote_c,
                                        "easiness": easiness, "n_distinct": n_distinct})
        item_tables[ds].to_csv(ROOT / f"results/tables/items_{ds}{args.suffix}.csv", index=False)
        s = summary["datasets"][ds]
        print(f"{ds:15s} best={s['best_acc']:.3f} vote={s['vote_acc']:.3f} diff={diff:+.3f} r={r:.3f} "
              f"AUC={auc:.3f} looAUC={s['loo_auc_mean']:.3f} phi={s['mean_error_phi']:.3f}")

    # Holm across datasets for McNemar
    ps = np.array([summary["datasets"][d]["mcnemar_p"] for d in DATASETS])
    for d, pa in zip(DATASETS, holm(ps)):
        summary["datasets"][d]["mcnemar_p_holm"] = float(pa)

    # domain-level pooled r / AUC (pool items of the two datasets in each domain)
    summary["domains"] = {}
    for dom in ["factual", "math", "ethics"]:
        t = pd.concat([item_tables[d] for d in DATASETS if DOMAIN[d] == dom])
        y, s_ = t.vote_correct.values, t.agreement.values
        summary["domains"][dom] = {"r": pb_r(y, s_), "auc": safe_auc(y, s_),
                                   "r_ci": boot_ci(lambda idx: pb_r(y[idx], s_[idx]), len(y), rng),
                                   "auc_ci": boot_ci(lambda idx: safe_auc(y[idx], s_[idx]), len(y), rng)}
    # bootstrap difference in r between domains
    for a, b in [("factual", "ethics"), ("math", "ethics")]:
        ta = pd.concat([item_tables[d] for d in DATASETS if DOMAIN[d] == a])
        tb = pd.concat([item_tables[d] for d in DATASETS if DOMAIN[d] == b])
        diffs = []
        for _ in range(B):
            ia = rng.integers(0, len(ta), len(ta)); ib = rng.integers(0, len(tb), len(tb))
            diffs.append(pb_r(ta.vote_correct.values[ia], ta.agreement.values[ia])
                         - pb_r(tb.vote_correct.values[ib], tb.agreement.values[ib]))
        diffs = np.array(diffs)
        summary["domains"][f"r_diff_{a}_minus_{b}"] = {"ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                                     "p_one_sided": float(np.mean(diffs <= 0))}

    json.dump(summary, open(ROOT / f"results/summary{args.suffix}.json", "w"), indent=2, default=float)
    make_tables(summary, args.suffix)
    if not args.suffix:
        make_figures(summary, item_tables, df, models)


def make_tables(S, suffix):
    rows = []
    for d in DATASETS:
        s = S["datasets"][d]
        rows.append({"dataset": d, "domain": s["domain"], "best model": f"{s['best_model']} ({s['best_acc']:.3f})",
                     "mean model acc": round(s["mean_model_acc"], 3), "vote acc": round(s["vote_acc"], 3),
                     "vote − best (95% CI)": f"{s['vote_minus_best']*100:+.1f} [{s['vote_minus_best_ci'][0]*100:+.1f}, {s['vote_minus_best_ci'][1]*100:+.1f}]",
                     "McNemar p (Holm)": f"{s['mcnemar_p_holm']:.3g}",
                     "vote − half-selected best": f"{s['split_vote_minus_selected_best']*100:+.1f}",
                     "5-subset vote − best (mean)": f"{s['sub5_vote_minus_best_mean']*100:+.1f}",
                     "r (95% CI)": f"{s['agree_r']:.3f} [{s['agree_r_ci'][0]:.2f}, {s['agree_r_ci'][1]:.2f}]",
                     "AUROC (95% CI)": f"{s['agree_auc']:.3f} [{s['agree_auc_ci'][0]:.2f}, {s['agree_auc_ci'][1]:.2f}]",
                     "LOO AUROC (mean)": round(s["loo_auc_mean"], 3),
                     "self-prob AUROC (mean)": round(s.get("selfprob_auc_mean", np.nan), 3),
                     "error φ": round(s["mean_error_phi"], 3),
                     "P(same wrong | both wrong)": round(s["mean_same_wrong_answer"], 3),
                     "unanimous %": round(100 * s["frac_unanimous"], 1),
                     "acc | unanimous": None if s["acc_when_unanimous"] is None else round(s["acc_when_unanimous"], 3),
                     "≥75% agree but wrong %": round(100 * s["frac_hi_agree_wrong"], 1)})
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / f"results/tables/main{suffix}.csv", index=False)
    open(ROOT / f"results/tables/main{suffix}.md", "w").write(t.to_markdown(index=False))
    acc = pd.DataFrame({d: S["datasets"][d]["model_acc"] for d in DATASETS})
    acc.loc["plurality vote"] = [S["datasets"][d]["vote_acc"] for d in DATASETS]
    open(ROOT / f"results/tables/accuracy{suffix}.md", "w").write(acc.round(3).to_markdown())
    print(acc.round(3))


def make_figures(S, item_tables, df, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"factual": "#2a6fdb", "math": "#1a9e77", "ethics": "#d95f02", "control": "#7570b3"}

    # Fig 1: accuracy of the plurality answer as a function of agreement level (calibration-style)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, dom in zip(axes, ["factual", "math", "ethics"]):
        if dom == "ethics" and "mmlu_tf" in DATASETS:  # overlay binary factual control on the ethics panel
            t = item_tables["mmlu_tf"]
            g = t.groupby(t.agreement.round(3)).agg(acc=("vote_correct", "mean"), n=("vote_correct", "size"))
            g = g[g.n >= 5]
            ax.plot(g.index, g.acc, "-.", marker="s", color=colors["control"], label="mmlu_tf (binary factual control)")
        for d, ls in zip([x for x in DATASETS if DOMAIN[x] == dom], ["-", "--"]):
            t = item_tables[d]
            g = t.groupby(t.agreement.round(3)).agg(acc=("vote_correct", "mean"), n=("vote_correct", "size"))
            g = g[g.n >= 5]
            ax.plot(g.index, g.acc, ls, marker="o", color=colors[dom], label=d)
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
        ax.set_title(dom); ax.set_xlabel("agreement rate (fraction of 8 models on plurality answer)")
        ax.legend(frameon=False, fontsize=8); ax.set_xlim(0.1, 1.02); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy of plurality answer")
    fig.suptitle("Plurality-answer accuracy vs inter-model agreement (bins with ≥5 items)")
    fig.tight_layout(); fig.savefig(ROOT / "figures/agreement_vs_accuracy.png", dpi=150); plt.close(fig)

    # Fig 2: r and AUROC with CIs per dataset
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    xs = np.arange(len(DATASETS))
    for ax, key, lab, band in [(axes[0], "agree_r", "point-biserial r", (0.4, 0.6)),
                               (axes[1], "agree_auc", "AUROC", (0.7, 0.8))]:
        v = [S["datasets"][d][key] for d in DATASETS]
        lo = [S["datasets"][d][key + "_ci"][0] for d in DATASETS]
        hi = [S["datasets"][d][key + "_ci"][1] for d in DATASETS]
        c = [colors[DOMAIN[d]] for d in DATASETS]
        ax.bar(xs, v, color=c, alpha=0.85)
        ax.errorbar(xs, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor="k", capsize=3)
        ax.axhspan(*band, color="grey", alpha=0.15, label="hypothesised range (factual/math)")
        if key == "agree_r":
            ax.axhline(0.2, color="#d95f02", ls="--", lw=1, label="hypothesised ethics ceiling (0.2)")
        else:
            ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(DATASETS, rotation=20); ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Agreement rate as an unsupervised correctness signal (95% bootstrap CI)")
    fig.tight_layout(); fig.savefig(ROOT / "figures/r_auc_by_dataset.png", dpi=150); plt.close(fig)

    # Fig 3: model accuracies vs vote
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.8 / (len(models) + 1)
    cmap = plt.get_cmap("tab10")
    for k, m in enumerate(models):
        ax.bar(xs + k * w, [S["datasets"][d]["model_acc"][m] for d in DATASETS], w, color=cmap(k), label=m)
    ax.bar(xs + len(models) * w, [S["datasets"][d]["vote_acc"] for d in DATASETS], w, color="k", label="plurality vote")
    ax.set_xticks(xs + 0.4); ax.set_xticklabels(DATASETS); ax.set_ylabel("accuracy")
    ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(ROOT / "figures/accuracy_models_vs_vote.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fig 3b: ensemble-size sweep
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for d, ls in zip(DATASETS, ["-", "--", "-", "--", "-", "--", "-."]):
        sw = S["datasets"][d]["size_sweep"]
        ks = sorted(int(k) for k in sw)
        axes[0].plot(ks, [sw[k]["auc"] for k in ks], ls, marker="o", color=colors[DOMAIN[d]], label=d)
        axes[1].plot(ks, [100 * sw[k]["vote_minus_best"] for k in ks], ls, marker="o", color=colors[DOMAIN[d]], label=d)
    axes[0].set_ylabel("AUROC of agreement"); axes[1].set_ylabel("vote − best-in-subset (pp)")
    axes[1].axhline(0, color="grey", lw=1)
    for ax in axes:
        ax.set_xlabel("number of models in ensemble"); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Effect of ensemble size (mean over model subsets)")
    fig.tight_layout(); fig.savefig(ROOT / "figures/ensemble_size_sweep.png", dpi=150); plt.close(fig)

    # Fig 4: pairwise error correlation heatmaps (factual vs ethics vs math)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.2 * len(DATASETS), 3.2))
    for ax, d in zip(axes, DATASETS):
        sub = df[df.dataset == d].pivot(index="id", columns="model", values="correct")[models].astype(float)
        E = 1 - sub.values
        M = np.corrcoef(E.T)
        ax.imshow(M, vmin=-0.2, vmax=1, cmap="viridis")
        ax.set_title(f"{d}\nmean φ={S['datasets'][d]['mean_error_phi']:.2f}", fontsize=9)
        ax.set_xticks(range(len(models))); ax.set_yticks(range(len(models)))
        ax.set_xticklabels(models, rotation=90, fontsize=6); ax.set_yticklabels(models if ax is axes[0] else [], fontsize=6)
    fig.suptitle("Pairwise correlation of error indicators", y=1.02)
    fig.tight_layout(); fig.savefig(ROOT / "figures/error_correlation.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
