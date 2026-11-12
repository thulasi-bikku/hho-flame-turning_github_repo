"""Reproduce every numerical result reported in the manuscript.

Usage:  python scripts/run_all.py            (full protocol, ~5-10 min)
        python scripts/run_all.py --quick    (smoke test)
Outputs are written to ./results
"""
import argparse, json, itertools, pathlib, sys, time
import numpy as np, pandas as pd
from scipy import stats, optimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flameopt import LB, UB, COEFFS, fitness, mrr, RA_MODELS, CONDITIONS          # noqa: E402
from flameopt.algorithms import run, ALGORITHMS                                   # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
ALGS = list(ALGORITHMS)                    # DE, WOA, CS, TLBO, HHO, PSO
N_POP, N_ITER = 30, 20                     # protocol of the manuscript
SUCCESS_TOL = 0.01                         # success: F <= F* + 0.01
TIE_DECIMALS = 6                           # fitness values are rounded to 1e-6 before ranking


def init_pop(run_id, N):
    return LB + np.random.default_rng(10_000 + run_id).random((N, 3)) * (UB - LB)


def one(alg, run_id, N=N_POP, T=N_ITER, max_fe=None, fun=fitness):
    ev = run(alg, fun, LB, UB, N=N, T=T, seed=20_000 + run_id, X0=init_pop(run_id, N), max_fe=max_fe)
    return ev


def fe_to_reach(marks, thr):
    for fe, b in marks:
        if b <= thr:
            return fe
    return np.nan


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run_max = 0
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * p[i]); adj[i] = min(1.0, run_max)
    return adj


def pairwise_tests(df, col="fit"):
    """Friedman omnibus + paired Wilcoxon signed-rank tests with Holm correction (runs share X0)."""
    wide = df.pivot(index="run", columns="alg", values=col)[ALGS].round(TIE_DECIMALS)   # differences < 1e-6 are ties
    chi, p_fr = stats.friedmanchisquare(*[wide[a] for a in ALGS])
    ranks = wide.rank(axis=1, method="average").mean()
    pairs, pv = list(itertools.combinations(ALGS, 2)), []
    for a, b in pairs:
        d = wide[a] - wide[b]
        pv.append(1.0 if np.allclose(d, 0) else stats.wilcoxon(wide[a], wide[b], zero_method="wilcox").pvalue)
    adj = holm(pv)
    pw = pd.DataFrame(index=ALGS, columns=ALGS, dtype=float)
    for (a, b), q in zip(pairs, adj):
        pw.loc[a, b] = pw.loc[b, a] = q
    return dict(friedman_chi2=float(chi), friedman_p=float(p_fr), mean_ranks=ranks.to_dict()), pw


def reference_optimum():
    best = None
    rng = np.random.default_rng(0)
    for _ in range(200):
        x0 = LB + rng.random(3) * (UB - LB)
        r = optimize.minimize(lambda x: fitness(x)[0], x0, bounds=list(zip(LB, UB)), method="L-BFGS-B",
                              options=dict(ftol=1e-15, gtol=1e-12))
        if best is None or r.fun < best.fun:
            best = r
    g = [np.linspace(l, u, 41) for l, u in zip(LB, UB)]
    G = np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T
    gf = fitness(G)
    return dict(x=best.x.tolist(), f=float(best.fun), grid_best_f=float(gf.min()), grid_best_x=G[gf.argmin()].tolist())


def done(name, force=False):
    return (OUT / name).exists() and not force


def main(quick=False, force=False):
    t0 = time.time()
    R = 8 if quick else 100
    summary = {}

    # ---- reference optimum ---------------------------------------------------------------------
    ref = reference_optimum(); summary["reference"] = ref
    fstar = ref["f"]; thr = fstar + SUCCESS_TOL
    print("reference optimum", ref)

    # ---- Protocol A: N=30, T=20 (iteration budget) ----------------------------------------------
    rows, hist = [], {a: [] for a in ALGS}
    for a in ([] if done("protocolA_runs.csv", force) else ALGS):
        for r in range(R):
            ev = one(a, r)
            rows.append(dict(alg=a, run=r, v=ev.best_x[0], f=ev.best_x[1], d=ev.best_x[2], fit=ev.best_f,
                             fe=ev.n, fe_success=fe_to_reach(ev.marks, thr)))
            hist[a].append([b for _, b in ev.marks])
    if rows:
        A = pd.DataFrame(rows); A.to_csv(OUT / "protocolA_runs.csv", index=False)
        np.savez(OUT / "protocolA_history.npz", **{a: np.array(h) for a, h in hist.items()})
    A = pd.read_csv(OUT / "protocolA_runs.csv")
    g = A.groupby("alg")
    tab = pd.DataFrame({
        "best": g.fit.min(), "worst": g.fit.max(), "mean": g.fit.mean(), "median": g.fit.median(),
        "std": g.fit.std(ddof=1), "success_rate": g.fit.apply(lambda s: (s <= thr).mean() * 100),
        "mean_fe": g.fe.mean(), "fe_to_success": g.fe_success.mean(),
        "v_mean": g.v.mean(), "v_std": g.v.std(ddof=1), "f_mean": g.f.mean(), "f_std": g.f.std(ddof=1),
        "d_mean": g.d.mean(), "d_std": g.d.std(ddof=1)}).loc[ALGS]
    tab.to_csv(OUT / "protocolA_summary.csv"); print(tab.round(4))
    omni, pw = pairwise_tests(A); summary["protocolA_tests"] = omni; pw.to_csv(OUT / "protocolA_pairwise_holm.csv")

    # ---- Protocol B: equal budget of 1,200 function evaluations ----------------------------------
    rows, histB = [], {a: [] for a in ALGS}
    for a in ([] if done("protocolB_runs.csv", force) else ALGS):
        for r in range(R):
            ev = one(a, r, T=10_000, max_fe=1200)
            rows.append(dict(alg=a, run=r, v=ev.best_x[0], f=ev.best_x[1], d=ev.best_x[2], fit=ev.best_f,
                             fe=ev.n, fe_success=fe_to_reach(ev.marks, thr)))
            m = np.array(ev.marks)
            grid = np.arange(30, 1201, 30)
            histB[a].append([m[m[:, 0] <= q, 1].min() if (m[:, 0] <= q).any() else np.nan for q in grid])
    if rows:
        B = pd.DataFrame(rows); B.to_csv(OUT / "protocolB_runs.csv", index=False)
        np.savez(OUT / "protocolB_history.npz", fe=np.arange(30, 1201, 30), **{a: np.array(h) for a, h in histB.items()})
    B = pd.read_csv(OUT / "protocolB_runs.csv")
    g = B.groupby("alg")
    tabB = pd.DataFrame({"best": g.fit.min(), "worst": g.fit.max(), "mean": g.fit.mean(), "median": g.fit.median(),
                         "std": g.fit.std(ddof=1),
                         "success_rate": g.fit.apply(lambda s: (s <= thr).mean() * 100),
                         "fe_to_success": g.fe_success.mean()}).loc[ALGS]
    tabB.to_csv(OUT / "protocolB_summary.csv"); print(tabB.round(4))
    omniB, pwB = pairwise_tests(B); summary["protocolB_tests"] = omniB; pwB.to_csv(OUT / "protocolB_pairwise_holm.csv")

    # ---- Iteration-budget study ------------------------------------------------------------------
    rows = []
    for T in (() if done("budget_iterations.csv", force) else (10, 20, 50, 100)):
        for a in ALGS:
            f = np.array([one(a, r, T=T).best_f for r in range(R)])
            rows.append(dict(T=T, alg=a, mean=f.mean(), std=f.std(ddof=1), success=(f <= thr).mean() * 100))
    if rows: pd.DataFrame(rows).to_csv(OUT / "budget_iterations.csv", index=False)

    # ---- Population-size study (T = 20) -----------------------------------------------------------
    rows = []
    for N in (() if done("budget_population.csv", force) else (10, 20, 30, 50, 100)):
        for a in ALGS:
            f = np.array([one(a, r, N=N).best_f for r in range(R)])
            rows.append(dict(N=N, alg=a, mean=f.mean(), std=f.std(ddof=1), success=(f <= thr).mean() * 100))
    if rows: pd.DataFrame(rows).to_csv(OUT / "budget_population.csv", index=False)

    # ---- Weight sensitivity of the composite objective (lambda multiplies 1/MRR) -----------------
    def solve(fun, T=100, N=30, reps=10, seed0=0):
        best = None
        for k in range(reps):
            ev = run("HHO", fun, LB, UB, N=N, T=T, seed=seed0 + k)
            if best is None or ev.best_f < best.best_f:
                best = ev
        return best
    rows = []
    for lam in (() if done("weight_sensitivity.csv", force) else (0.25, 0.5, 1, 2, 4, 8)):
        ev = solve(lambda X, lam=lam: fitness(X, lam=lam))
        x = ev.best_x
        rows.append(dict(lam=lam, v=x[0], f=x[1], d=x[2], F=ev.best_f, MRR=float(mrr(x)[0]),
                         Ra_dry=float(RA_MODELS["dry"](x)[0]), Ra_carb=float(RA_MODELS["carburizing"](x)[0]),
                         Ra_ox=float(RA_MODELS["oxidizing"](x)[0])))
    if rows: pd.DataFrame(rows).to_csv(OUT / "weight_sensitivity.csv", index=False)

    # ---- Per-condition bi-objective analysis (Ra_k vs MRR) ----------------------------------------
    g3 = [np.linspace(l, u, 61) for l, u in zip(LB, UB)]
    G = np.array(np.meshgrid(*g3, indexing="ij")).reshape(3, -1).T
    M = mrr(G)
    wrows, pareto = [], {}
    for k in CONDITIONS:
        if done("weighted_sum_by_condition.csv", force): break
        Ra = RA_MODELS[k](G)
        lo, hi, mlo, mhi = Ra.min(), Ra.max(), M.min(), M.max()
        order = np.argsort(Ra); front, best_m = [], -np.inf          # exact non-dominated set on the grid
        for i in order:
            if M[i] > best_m:
                front.append(i); best_m = M[i]
        pareto[k] = np.c_[Ra[front], M[front], G[front]]

        def fun(X, k=k, lo=lo, hi=hi, mlo=mlo, mhi=mhi, w=None):
            raise NotImplementedError
        for w in np.round(np.linspace(0.05, 0.95, 19), 2):
            f_w = lambda X, k=k, lo=lo, hi=hi, mlo=mlo, mhi=mhi, w=w: (
                w * (RA_MODELS[k](X) - lo) / (hi - lo) + (1 - w) * (mhi - mrr(X)) / (mhi - mlo))
            ev = solve(f_w, T=50, reps=5)
            x = ev.best_x
            wrows.append(dict(condition=k, w=w, v=x[0], f=x[1], d=x[2], Ra=float(RA_MODELS[k](x)[0]),
                              MRR=float(mrr(x)[0])))
    if wrows:
        pd.DataFrame(wrows).to_csv(OUT / "weighted_sum_by_condition.csv", index=False)
        np.savez(OUT / "pareto_grid.npz", **pareto)

    # ---- Parameter-uncertainty (Monte Carlo) analysis ---------------------------------------------
    rows, rng = [], np.random.default_rng(42)
    for rel in (() if done("monte_carlo_optimum.csv", force) else (0.05, 0.10)):
        for s in range(20 if quick else 500):
            c = {k: tuple(np.array(v) * (1 + rng.uniform(-rel, rel, 4))) for k, v in COEFFS.items()}
            r = optimize.differential_evolution(lambda x: fitness(x, coeffs=c)[0], list(zip(LB, UB)),
                                                seed=s, tol=1e-10, maxiter=300, polish=True)
            rows.append(dict(rel=rel, sample=s, v=r.x[0], f=r.x[1], d=r.x[2], F=r.fun))
    if rows: pd.DataFrame(rows).to_csv(OUT / "monte_carlo_optimum.csv", index=False)

    # ---- Model predictions on the 3^3 grid and marginal means -------------------------------------
    lv = {"v": [34, 64, 94], "f": [0.113, 0.178, 0.249], "d": [0.25, 0.5, 0.75]}
    full = np.array(list(itertools.product(lv["v"], lv["f"], lv["d"])), float)
    grid = pd.DataFrame(full, columns=["v", "f", "d"])
    for k in CONDITIONS:
        grid["Ra_" + k] = RA_MODELS[k](full)
    grid["MRR"] = mrr(full); grid.to_csv(OUT / "model_predictions_27_grid.csv", index=False)
    marg = []
    for j, p in enumerate(("v", "f", "d")):
        for lvl in lv[p]:
            sel = grid[p] == lvl
            marg.append(dict(parameter=p, level=lvl, **{k: grid.loc[sel, "Ra_" + k].mean() for k in CONDITIONS}))
    marg = pd.DataFrame(marg)
    for k in ("carburizing", "oxidizing"):
        marg["chg_" + k] = (marg[k] / marg["dry"] - 1) * 100
    marg.to_csv(OUT / "marginal_means.csv", index=False)
    overall = {k: grid["Ra_" + k].mean() for k in CONDITIONS}
    summary["overall_mean_Ra"] = overall
    summary["overall_change_pct"] = {k: (overall[k] / overall["dry"] - 1) * 100 for k in ("carburizing", "oxidizing")}
    # exact variance decomposition of ln(Ra) on the factorial grid
    contrib = {}
    for k in CONDITIONS:
        _, a, b, e = COEFFS[k]
        comp = [a ** 2 * np.var(np.log(lv["v"])), b ** 2 * np.var(np.log(lv["f"])), e ** 2 * np.var(np.log(lv["d"]))]
        contrib[k] = (np.array(comp) / sum(comp) * 100).tolist()
    summary["contribution_pct_v_f_d"] = contrib
    summary["Ra_range_grid"] = {k: [float(grid["Ra_" + k].min()), float(grid["Ra_" + k].max())] for k in CONDITIONS}
    summary["runtime_s"] = time.time() - t0
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print("finished in %.1f s" % summary["runtime_s"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute stages even if outputs exist")
    a = ap.parse_args(); main(a.quick, a.force)
