"""Create all manuscript figures from ./results (run scripts/run_all.py first)."""
import pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flameopt import LB, UB, RA_MODELS, fitness, mrr                              # noqa: E402
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150,
                     "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
COND = ["dry", "carburizing", "oxidizing"]
LAB = {"dry": "Dry", "carburizing": "Carburizing flame", "oxidizing": "Oxidizing flame"}
STY = {"dry": ("k", "o", "-"), "carburizing": ("tab:blue", "s", "--"), "oxidizing": ("tab:red", "^", ":")}
ALG = ["DE", "WOA", "CS", "TLBO", "HHO", "PSO"]
ACOL = dict(zip(ALG, ["tab:gray", "tab:cyan", "tab:green", "tab:orange", "tab:red", "tab:purple"]))
AMK = dict(zip(ALG, ["o", "s", "^", "D", "v", "P"]))
FSTAR = 3.403388317104343


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white"); plt.close(fig)


def box(ax, xy, w, h, text, fc="#e8eef7"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec="k", lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8)


def fig1():
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    items = [((0.2, 3.2), "Experimental campaign\nand regression models\n(authors' earlier work)", "#f3dede"),
             ((3.7, 3.2), "Composite objective F\n= ΣRa + 1/MRR with\nbounds on v, f, d", "#e6e0f0"),
             ((7.2, 3.2), "Six metaheuristics\nDE, WOA, CS, TLBO,\nHHO, PSO", "#dcebf5"),
             ((7.2, 0.9), "Statistical comparison\n100 runs, two budgets\n(iterations and FE)", "#e4efd9"),
             ((3.7, 0.9), "Extended analyses:\nweights, Pareto fronts,\nparameter uncertainty", "#f7ecd2"),
             ((0.2, 0.9), "Recommended cutting\nparameters and\nprocess implications", "#fbe3cf")]
    for xy, t, c in items:
        box(ax, xy, 2.6, 1.3, t, c)
    for a, b in [((2.85, 3.85), (3.65, 3.85)), ((6.35, 3.85), (7.15, 3.85)), ((8.5, 3.15), (8.5, 2.25)),
                 ((7.15, 1.55), (6.35, 1.55)), ((3.65, 1.55), (2.85, 1.55))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1))
    save(fig, "fig01_workflow.png")


def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1.8, 1]})
    a.set_aspect("equal"); a.axis("off"); a.set_xlim(-1, 21); a.set_ylim(-5, 5)
    a.add_patch(Rectangle((0, -3.75), 20, 7.5, fc="#d9d9d9", ec="k"))
    for x in np.linspace(0.8, 17.5, 8):
        a.add_patch(Rectangle((x, -3.75), 1.2, 7.5, fc="#bdbdbd", ec="k", lw=0.5))
    a.text(10, 0, "Bar stock, Ø75 mm × 300 mm", ha="center", va="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=2))
    a.annotate("", xy=(0.8, -4.4), xytext=(2.0, -4.4), arrowprops=dict(arrowstyle="<->"))
    a.text(1.4, -5.0, "10–15 mm\nper test", ha="center", va="top", fontsize=7)
    a.plot([13.5, 13.5], [3.75, 4.9], "k", lw=1.2); a.text(14.0, 4.3, "stylus", fontsize=7)
    a.set_title("(a) Side view (schematic, not to scale)", fontsize=9)
    b.set_aspect("equal"); b.axis("off"); b.set_xlim(-1.8, 1.8); b.set_ylim(-1.8, 1.8)
    b.add_patch(Circle((0, 0), 1.3, fc="#d9d9d9", ec="k"))
    for ang in (90, 210, 330):
        t = np.deg2rad(ang); b.plot(1.3 * np.cos(t), 1.3 * np.sin(t), "ro", ms=5)
        b.plot([0, 1.3 * np.cos(t)], [0, 1.3 * np.sin(t)], "r--", lw=0.6)
    b.text(0, -1.75, "Three measurement points,\n120° apart", ha="center", fontsize=7)
    b.set_title("(b) End view", fontsize=9)
    save(fig, "fig02_measurement_schematic.png")


def fig3():
    fig, ax = plt.subplots(figsize=(5.2, 4)); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-3.2, 3.6)
    ax.add_patch(Circle((0, 0), 2, fc="#d9d9d9", ec="k"))
    ax.annotate("", xy=(0, 1.0), xytext=(0.8, 0.8), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.5"))
    ax.text(0, 0, "Rotation", ha="center", fontsize=7)
    # torch at upper-left, flame towards surface
    ax.add_patch(Rectangle((-3.9, 2.4), 1.5, 0.35, angle=-35, fc="#555555", ec="k"))
    ax.annotate("", xy=(-1.45, 1.4), xytext=(-2.45, 2.0), arrowprops=dict(arrowstyle="-|>", color="orange", lw=3))
    ax.text(-3.9, 3.2, "Oxy-acetylene torch\n(carburizing: C₂H₂-rich 2.5:1;\noxidizing: O₂-rich 2.5:1)", fontsize=7, va="center")
    ax.text(-3.5, 0.9, "4 cm gap", fontsize=7, color="darkorange")
    # water spray
    ax.annotate("", xy=(0.2, 1.95), xytext=(-0.6, 3.0), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2, ls="--"))
    ax.text(-0.5, 3.15, "Water spray (quench)", fontsize=7, color="tab:blue")
    # tool at right
    ax.add_patch(Rectangle((2.0, -0.25), 1.8, 0.5, fc="#8c8c8c", ec="k"))
    ax.text(2.9, -0.7, "Insert and holder", ha="center", fontsize=7)
    ax.annotate("", xy=(2.0, -1.4), xytext=(0.8, -1.9), arrowprops=dict(arrowstyle="<-"))
    ax.text(-0.1, -2.5, "Feed direction", fontsize=7)
    th = np.linspace(np.deg2rad(125), np.deg2rad(20), 40)
    ax.plot(2.25 * np.cos(th), 2.25 * np.sin(th), "k:", lw=0.8)
    ax.text(1.7, 2.25, "≈5 cm arc, flame to tool", fontsize=7)
    ax.text(0, -3.1, "Schematic, not to scale", ha="center", fontsize=7, style="italic")
    save(fig, "fig03_flame_setup_schematic.png")


def marg():
    return pd.read_csv(RES / "marginal_means.csv")


def fig4_5():
    m = marg()
    for fname, params, xl in [("fig04_ra_vs_speed.png", [("v", "Cutting speed, v (m/min)")], None),
                              ("fig05_ra_vs_feed_depth.png", [("f", "Feed, f (mm/rev)"), ("d", "Depth of cut, d (mm)")], None)]:
        fig, axs = plt.subplots(1, len(params), figsize=(3.3 * len(params) + (0.4 if len(params) == 1 else 0), 2.9), squeeze=False)
        for ax, (p, xlab) in zip(axs[0], params):
            s = m[m.parameter == p]
            for k in COND:
                c, mk, ls = STY[k]; ax.plot(s.level, s[k], color=c, marker=mk, ls=ls, label=LAB[k])
            ax.set_xlabel(xlab); ax.set_ylabel("Mean model-predicted Ra (µm)"); ax.set_xticks(s.level)
        axs[0][0].legend(frameon=False)
        if len(params) == 2:
            axs[0][0].set_title("(a)", loc="left"); axs[0][1].set_title("(b)", loc="left")
        fig.tight_layout(); save(fig, fname)


def fig6():
    v = np.linspace(LB[0], UB[0], 120); f = np.linspace(LB[1], UB[1], 120); V, F = np.meshgrid(v, f)
    X = np.c_[V.ravel(), F.ravel(), np.full(V.size, 0.5)]
    Z = {k: RA_MODELS[k](X).reshape(V.shape) for k in COND}
    lo, hi = min(z.min() for z in Z.values()), max(z.max() for z in Z.values())
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
    for ax, k, tag in zip(axs, COND, "abc"):
        cs = ax.contourf(V, F, Z[k], levels=np.linspace(lo, hi, 13), cmap="viridis")
        ax.set_title(f"({tag}) {LAB[k]}", fontsize=8); ax.set_xlabel("v (m/min)"); ax.grid(False)
    axs[0].set_ylabel("f (mm/rev)")
    fig.colorbar(cs, ax=axs, label="Ra (µm), d = 0.5 mm", shrink=0.9, pad=0.02)
    save(fig, "fig06_ra_contours.png")


def fig7_8():
    H = np.load(RES / "protocolA_history.npz")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for a in ALG:
        y = np.maximum(H[a].mean(axis=0) - FSTAR, 1e-9)
        ax.semilogy(np.arange(len(y)), y, color=ACOL[a], marker=AMK[a], ms=3.5, label=a)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean (F − F*) over 100 runs"); ax.legend(ncol=3, frameon=False)
    save(fig, "fig07_convergence_iterations.png")
    HB = np.load(RES / "protocolB_history.npz"); fe = HB["fe"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for a in ALG:
        y = np.maximum(np.nanmean(HB[a], axis=0) - FSTAR, 1e-9)
        ax.semilogy(fe, y, color=ACOL[a], label=a)
    ax.set_xlabel("Function evaluations"); ax.set_ylabel("Mean (F − F*) over 100 runs"); ax.legend(ncol=3, frameon=False)
    save(fig, "fig08_convergence_fe.png")


def fig9_10():
    A = pd.read_csv(RES / "protocolA_runs.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    data = [np.maximum(A[A.alg == a].fit.values - FSTAR, 0) for a in ALG]
    ax.boxplot(data, tick_labels=ALG, showfliers=True, flierprops=dict(marker="o", ms=3, mfc="none"))
    ax.set_yscale("symlog", linthresh=1e-4); ax.set_ylabel("Final F − F* (symlog scale)")
    save(fig, "fig09_boxplot.png")
    fig, ax = plt.subplots(figsize=(4.8, 3.2)); rng = np.random.default_rng(0)
    for i, a in enumerate(ALG):
        d = A[A.alg == a].d.values; ax.plot(i + rng.uniform(-0.22, 0.22, len(d)), d, ".", color=ACOL[a], ms=4, alpha=0.7)
    ax.axhline(0.3938, color="k", ls="--", lw=0.8); ax.text(5.45, 0.40, "d* = 0.394", ha="right", fontsize=7)
    ax.set_xticks(range(len(ALG))); ax.set_xticklabels(ALG); ax.set_ylabel("Final depth of cut, d (mm)")
    save(fig, "fig10_depth_spread.png")


def fig11():
    P = np.load(RES / "pareto_grid.npz"); W = pd.read_csv(RES / "weighted_sum_by_condition.csv")
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6))
    for ax, k, tag in zip(axs, COND, "abc"):
        p = P[k]; ax.plot(p[:, 1], p[:, 0], "-", color="0.35", lw=1, label="Pareto front (grid)")
        w = W[W.condition == k]; ax.plot(w.MRR, w.Ra, "o", color="tab:red", ms=4, mfc="none", label="HHO weighted-sum")
        ax.set_title(f"({tag}) {LAB[k]}", fontsize=8); ax.set_xlabel("MRR (cm³/min)")
    axs[0].set_ylabel("Ra (µm)"); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); save(fig, "fig11_pareto.png")


def fig12_13():
    S = pd.read_csv(RES / "weight_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.semilogx(S.lam, S.d, "o-", color="tab:red"); ax.set_xticks(S.lam); ax.set_xticklabels([str(x) for x in S.lam])
    ax.set_xlabel("Weight λ on 1/MRR"); ax.set_ylabel("Optimal depth of cut, d* (mm)"); ax.minorticks_off()
    save(fig, "fig12_weight_sensitivity.png")
    M = pd.read_csv(RES / "monte_carlo_optimum.csv")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for rel, c in [(0.05, "tab:blue"), (0.10, "tab:orange")]:
        s = M[M.rel == rel]
        axs[0].hist(s.d, bins=np.linspace(0.25, 0.75, 26), alpha=0.6, color=c, label=f"±{int(rel*100)}%")
        axs[1].hist(s.F, bins=25, alpha=0.6, color=c, label=f"±{int(rel*100)}%")
    axs[0].set_xlabel("Optimal depth of cut, d* (mm)"); axs[0].set_ylabel("Samples"); axs[0].legend(frameon=False)
    axs[1].set_xlabel("Minimum objective, F*"); axs[0].set_title("(a)", loc="left"); axs[1].set_title("(b)", loc="left")
    fig.tight_layout(); save(fig, "fig13_parameter_uncertainty.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4_5(); fig6(); fig7_8(); fig9_10()
    if (RES / "weighted_sum_by_condition.csv").exists():
        fig11()
    if (RES / "monte_carlo_optimum.csv").exists():
        fig12_13()
    print("figures written to", FIG)
