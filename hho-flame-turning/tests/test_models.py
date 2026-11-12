import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from flameopt import LB, UB, fitness, mrr, ra_carburizing
from flameopt.algorithms import run, ALGORITHMS


def test_reference_value():
    # best objective value reported in the manuscript: v=94, f=0.113, d=0.394 -> F = 3.4034
    assert abs(fitness([94, 0.113, 0.394])[0] - 3.4034) < 5e-4


def test_mrr_units():
    # 94 m/min * 0.113 mm/rev * 0.394 mm = 4.185 cm^3/min
    assert abs(mrr([94, 0.113, 0.394])[0] - 4.185) < 1e-3


def test_carburizing_below_dry_at_optimum():
    from flameopt import ra_dry
    x = [94, 0.113, 0.394]
    assert ra_carburizing(x)[0] < ra_dry(x)[0]


def test_algorithms_respect_bounds_and_reach_optimum():
    for name in ALGORITHMS:
        ev = run(name, fitness, LB, UB, N=30, T=60, seed=3)
        assert np.all(ev.best_x >= LB - 1e-12) and np.all(ev.best_x <= UB + 1e-12)
        assert ev.best_f < 3.41, name


def test_budget_is_enforced():
    ev = run("TLBO", fitness, LB, UB, N=30, T=10_000, seed=1, max_fe=300)
    assert ev.n <= 300
