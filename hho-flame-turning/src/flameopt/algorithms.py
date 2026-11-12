"""Six population-based metaheuristics (DE, WOA, CS, TLBO, HHO, PSO).

All algorithms
  * minimize a vectorized objective ``fun(X) -> (n,)`` over box bounds [lb, ub],
  * clip every candidate to the bounds before evaluation,
  * share an evaluation counter, so results can be compared per iteration or per
    function evaluation (FE),
  * accept a common initial population ``X0`` (common random numbers across algorithms).

References: Storn & Price (1997) DE; Mirjalili & Lewis (2016) WOA; Yang & Deb (2009) CS;
Rao et al. (2011) TLBO; Heidari et al. (2019) HHO; Kennedy & Eberhart (1995) PSO.
"""
from math import gamma, pi, sin
import numpy as np


class BudgetExhausted(Exception):
    """Raised when an evaluation request would exceed the FE budget."""


class Evaluator:
    """Wraps the objective, counts evaluations, tracks the best-so-far solution."""

    def __init__(self, fun, budget=None):
        self.fun, self.budget = fun, budget
        self.n = 0
        self.best_f, self.best_x = np.inf, None
        self.marks = []          # (cumulative FE, best-so-far) at the end of every iteration

    def __call__(self, X):
        X = np.atleast_2d(X)
        if self.budget is not None and self.n + len(X) > self.budget:
            raise BudgetExhausted
        f = np.asarray(self.fun(X), dtype=float)
        self.n += len(X)
        i = int(np.argmin(f))
        if f[i] < self.best_f:
            self.best_f, self.best_x = float(f[i]), X[i].copy()
        return f

    def mark(self):
        self.marks.append((self.n, self.best_f))


def _levy(rng, shape, beta=1.5):
    sigma = (gamma(1 + beta) * sin(pi * beta / 2)
             / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, shape)
    v = rng.normal(0, 1, shape)
    return u / np.abs(v) ** (1 / beta)


def _init(rng, lb, ub, N, X0):
    if X0 is not None:
        return np.clip(np.array(X0, dtype=float), lb, ub)
    return lb + rng.random((N, len(lb))) * (ub - lb)


# ----------------------------------------------------------------------------------------
def de(ev, lb, ub, N, T, rng, X0=None, F=0.8, CR=0.9):
    """DE/rand/1/bin."""
    D = len(lb)
    X = _init(rng, lb, ub, N, X0)
    fx = ev(X)
    ev.mark()
    for _ in range(T):
        R = rng.random((N, N))
        np.fill_diagonal(R, np.inf)
        idx = np.argsort(R, axis=1)[:, :3]
        V = X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]])
        cross = rng.random((N, D)) < CR
        cross[np.arange(N), rng.integers(0, D, N)] = True
        U = np.clip(np.where(cross, V, X), lb, ub)
        fu = ev(U)
        better = fu < fx
        X[better], fx[better] = U[better], fu[better]
        ev.mark()


def woa(ev, lb, ub, N, T, rng, X0=None, b=1.0):
    D = len(lb)
    X = _init(rng, lb, ub, N, X0)
    fx = ev(X)
    leader, fl = X[np.argmin(fx)].copy(), fx.min()
    ev.mark()
    for t in range(T):
        a = 2 - 2 * t / T
        r1, r2 = rng.random(N), rng.random(N)
        A = (2 * a * r1 - a)[:, None]
        C = (2 * r2)[:, None]
        p = rng.random(N)[:, None]
        l = (rng.random(N) * 2 - 1)[:, None]
        Xrand = X[rng.integers(0, N, N)]
        encircle = np.where(np.abs(A) >= 1,
                            Xrand - A * np.abs(C * Xrand - X),
                            leader - A * np.abs(C * leader - X))
        spiral = np.abs(leader - X) * np.exp(b * l) * np.cos(2 * pi * l) + leader
        X = np.clip(np.where(p < 0.5, encircle, spiral), lb, ub)
        fx = ev(X)
        if fx.min() < fl:
            leader, fl = X[np.argmin(fx)].copy(), fx.min()
        ev.mark()


def cs(ev, lb, ub, N, T, rng, X0=None, pa=0.25, beta=1.5):
    D = len(lb)
    X = _init(rng, lb, ub, N, X0)
    fx = ev(X)
    ev.mark()
    for _ in range(T):
        best = X[np.argmin(fx)]
        step = _levy(rng, (N, D), beta)
        new = np.clip(X + 0.01 * step * (X - best) * rng.normal(size=(N, D)), lb, ub)
        fn = ev(new)
        m = fn < fx
        X[m], fx[m] = new[m], fn[m]
        K = rng.random((N, D)) < pa
        new = np.clip(X + rng.random((N, 1)) * (X[rng.permutation(N)] - X[rng.permutation(N)]) * K, lb, ub)
        fn = ev(new)
        m = fn < fx
        X[m], fx[m] = new[m], fn[m]
        ev.mark()


def tlbo(ev, lb, ub, N, T, rng, X0=None):
    D = len(lb)
    X = _init(rng, lb, ub, N, X0)
    fx = ev(X)
    ev.mark()
    for _ in range(T):
        teacher = X[np.argmin(fx)].copy()
        TF = rng.integers(1, 3, (N, 1))
        new = np.clip(X + rng.random((N, D)) * (teacher - TF * X.mean(axis=0)), lb, ub)
        fn = ev(new)
        m = fn < fx
        X[m], fx[m] = new[m], fn[m]
        J = (np.arange(N) + rng.integers(1, N, N)) % N        # partner j != i
        sign = np.where(fx < fx[J], 1.0, -1.0)[:, None]
        new = np.clip(X + rng.random((N, D)) * sign * (X - X[J]), lb, ub)
        fn = ev(new)
        m = fn < fx
        X[m], fx[m] = new[m], fn[m]
        ev.mark()


def hho(ev, lb, ub, N, T, rng, X0=None, beta=1.5):
    """Harris hawks optimization (Heidari et al., 2019), all four besiege strategies."""
    D = len(lb)
    X = _init(rng, lb, ub, N, X0)
    fx = ev(X)
    rabbit, fr = X[np.argmin(fx)].copy(), fx.min()
    ev.mark()
    for t in range(T):
        E = 2 * (1 - t / T) * (2 * rng.random(N) - 1)
        q, r = rng.random(N), rng.random(N)
        J = 2 * (1 - rng.random(N))[:, None]
        Xm = X.mean(axis=0)
        Xnew = X.copy()
        for i in range(N):
            Ei = E[i]
            if abs(Ei) >= 1:                                   # exploration
                if q[i] >= 0.5:
                    Xr = X[rng.integers(0, N)]
                    Xnew[i] = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * X[i])
                else:
                    Xnew[i] = (rabbit - Xm) - rng.random() * (lb + rng.random() * (ub - lb))
            elif r[i] >= 0.5 and abs(Ei) >= 0.5:               # soft besiege
                Xnew[i] = (rabbit - X[i]) - Ei * np.abs(J[i] * rabbit - X[i])
            elif r[i] >= 0.5:                                  # hard besiege
                Xnew[i] = rabbit - Ei * np.abs(rabbit - X[i])
            else:                                              # besiege with progressive rapid dives
                ref = X[i] if abs(Ei) >= 0.5 else Xm
                Y = np.clip(rabbit - Ei * np.abs(J[i] * rabbit - ref), lb, ub)
                if ev(Y)[0] < fx[i]:
                    Xnew[i] = Y
                else:
                    Z = np.clip(Y + rng.random(D) * _levy(rng, D, beta), lb, ub)
                    Xnew[i] = Z if ev(Z)[0] < fx[i] else X[i]
        X = np.clip(Xnew, lb, ub)
        fx = ev(X)
        if fx.min() < fr:
            rabbit, fr = X[np.argmin(fx)].copy(), fx.min()
        ev.mark()


def pso(ev, lb, ub, N, T, rng, X0=None, c1=2.0, c2=2.0, w0=0.9, w1=0.4):
    D = len(lb)
    X = _init(rng, lb, ub, N, X0)
    vmax = 0.2 * (ub - lb)
    V = rng.uniform(-vmax, vmax, (N, D))
    fx = ev(X)
    P, fp = X.copy(), fx.copy()
    g = P[np.argmin(fp)].copy()
    ev.mark()
    for t in range(T):
        w = w0 - (w0 - w1) * t / max(T - 1, 1)
        V = np.clip(w * V + c1 * rng.random((N, D)) * (P - X) + c2 * rng.random((N, D)) * (g - X), -vmax, vmax)
        X = np.clip(X + V, lb, ub)
        fx = ev(X)
        m = fx < fp
        P[m], fp[m] = X[m], fx[m]
        g = P[np.argmin(fp)].copy()
        ev.mark()


ALGORITHMS = {"DE": de, "WOA": woa, "CS": cs, "TLBO": tlbo, "HHO": hho, "PSO": pso}


def run(name, fun, lb, ub, N=30, T=20, seed=0, X0=None, max_fe=None):
    """Run one algorithm. Iteration budget T, or an FE budget ``max_fe`` (T then acts as a cap)."""
    ev = Evaluator(fun, budget=max_fe)
    rng = np.random.default_rng(seed)
    try:
        ALGORITHMS[name](ev, np.asarray(lb, float), np.asarray(ub, float), N, T, rng, X0=X0)
    except BudgetExhausted:
        pass
    return ev
