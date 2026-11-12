"""Empirical power-law models and the optimization objective.

Units: v in m/min, f in mm/rev, d in mm, Ra in micrometres.
MRR = v * f * d is expressed in cm^3/min (1 m/min = 1000 mm/min, so
v*1000*f*d mm^3/min = v*f*d cm^3/min).
"""
import numpy as np

LB = np.array([34.0, 0.113, 0.25])   # lower bounds of (v, f, d)
UB = np.array([94.0, 0.249, 0.75])   # upper bounds of (v, f, d)

# (constant, exponent of v, exponent of f, exponent of d)
COEFFS = {
    "carburizing": (22.1005, -0.052, 1.32, 0.06),
    "oxidizing":   (81.45,   -0.356, 1.20, 0.106),
    "dry":         (22.83,   -0.028, 1.28, 0.06),
}
CONDITIONS = ("dry", "carburizing", "oxidizing")


def _power_law(X, c):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    k, a, b, e = c
    return k * X[:, 0] ** a * X[:, 1] ** b * X[:, 2] ** e


def ra_carburizing(X, coeffs=None):
    return _power_law(X, (coeffs or COEFFS)["carburizing"])


def ra_oxidizing(X, coeffs=None):
    return _power_law(X, (coeffs or COEFFS)["oxidizing"])


def ra_dry(X, coeffs=None):
    return _power_law(X, (coeffs or COEFFS)["dry"])


RA_MODELS = {"dry": ra_dry, "carburizing": ra_carburizing, "oxidizing": ra_oxidizing}


def mrr(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return X[:, 0] * X[:, 1] * X[:, 2]


def fitness(X, coeffs=None, lam=1.0):
    """Composite objective: Ra(carburizing) + Ra(oxidizing) + Ra(dry) + lam / MRR."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return (ra_carburizing(X, coeffs) + ra_oxidizing(X, coeffs) + ra_dry(X, coeffs)
            + lam / mrr(X))
