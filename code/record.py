"""Recompute the monotone towers' good-coordinate values, in double precision and with 40-digit mpmath.

Gates are monotone Boolean functions of z in {0,1}^k given by positive DNFs:
  M      perfect matching of the 3x3 matrix (K_{3,3}),  D = NOT M(NOT z)
  P      perfect matching of the 4x4 matrix
  T(w,s) OR of s disjoint ANDs of w variables (tribes), T_s = T(2,s);  Q_s = dual of T_s
  PM(K4) OR of the three pairs of opposite edges of K4, i.e. T_3 up to relabelling
In +-1 form a variable is x = -1 when z = 1 and the output is -1 when the gate is true; this is the convention
that reproduces the claimed M/D tower values (convention_scan.py).  Words are read bottom to top: the first gate
is applied to single variables.  Every gate is transitive on its inputs, so leaf position 0 represents every leaf.
"""
import itertools
import sys
import mpmath as mp
import numpy as np

mp.mp.dps = 40


def tt_from(fn, k):
    return [(-1 if fn([(y >> j) & 1 for j in range(k)]) else 1) for y in range(2 ** k)]


def dual(fn):
    return lambda z: not fn([1 - v for v in z])


def pm(nside):
    perms = list(itertools.permutations(range(nside)))
    return lambda z: any(all(z[nside * u + p[u]] for u in range(nside)) for p in perms)


def tribes(w, s):
    return lambda z: any(all(z[w * j + l] for l in range(w)) for j in range(s))


def pm_k4(z):  # edges 01,23 | 02,13 | 03,12 as variables (0,1),(2,3),(4,5)
    return (z[0] and z[1]) or (z[2] and z[3]) or (z[4] and z[5])


GATES = {
    "M": (pm(3), 9), "D": (dual(pm(3)), 9), "P": (pm(4), 16),
    "dP": (dual(pm(4)), 16), "Q2": (dual(tribes(2, 2)), 4), "dPMK4": (dual(pm_k4), 6),
    "dT43": (dual(tribes(4, 3)), 12), "T26": (tribes(2, 6), 12), "T35": (tribes(3, 5), 15),
}


class Arith:
    def __init__(self, prec):
        self.prec = prec
        if prec == "mp":
            self.f = mp.mpf
            self.log2 = lambda v: mp.log(v, 2)
            self.sqrt = mp.sqrt
        else:
            self.f = float
            self.log2 = lambda v: float(np.log2(v))
            self.sqrt = lambda v: float(np.sqrt(v))

    def m(self, a, b):
        s = a + b
        out = 0
        if a > 0:
            out += a * self.log2(s / a)
        if b > 0:
            out += b * self.log2(s / b)
        return out


def biased_coeffs(tt, k, mu, A):
    """Butterfly: c[T] = E[g(y) prod_{j in T} (y_j - mu)/sigma], y_j iid of mean mu; bit j of y set <=> y_j = -1."""
    one = A.f(1)
    sig = A.sqrt(one - mu * mu)
    pp, pm_ = (one + mu) / 2, (one - mu) / 2
    fp, fm = (one - mu) / sig, (-one - mu) / sig
    c = [A.f(v) for v in tt]
    h = 1
    while h < len(c):
        for i in range(0, len(c), 2 * h):
            for t in range(i, i + h):
                a, b = c[t], c[t + h]          # a: y_j = +1, b: y_j = -1
                c[t] = pp * a + pm_ * b
                c[t + h] = pp * fp * a + pm_ * fm * b
        h *= 2
    return c


def step(state, tt, k, A):
    mu, P1, e, Inf = state
    s2 = A.f(1) - mu * mu
    c = biased_coeffs(tt, k, mu, A)
    c2 = [v * v for v in c]
    W = sum(c2[T] for T in range(2 ** k) if T & 1)
    e_new = W / s2 * e
    for T in range(2, 2 ** k, 2):
        e_new += A.m(c2[T], c2[T | 1] * P1 / s2)
    return (c[0], c2[1] * P1 / s2, e_new, W * Inf / s2)


def tower(word, A):
    st = (A.f(0), A.f(1), A.f(0), A.f(1))
    tts = {}
    for g in word:
        if g not in tts:
            fn, k = GATES[g]
            tts[g] = (tt_from(fn, k), k)
        tt, k = tts[g]
        st = step(st, tt, k, A)
    mu, P1, e, Inf = st
    E = e + A.m(mu * mu, P1)
    return E / Inf, st


WORDS = {
    "old PM record M,D,D,M,M,D,D": (["M", "D", "D", "M", "M", "D", "D"], "4.50504996749762792695710133416697266496"),
    "plain M/D word MDDDDDMM": (list("MDDDDDMM"), "4.506379879463887"),
    "small witness M,D,D,D,D,dual(T(2,5))": None,
    "B14": (["M", "D", "D", "D", "D", "M", "P", "P", "P", "Q2", "dPMK4", "dT43", "T26", "T35"],
                      "4.5064006455155684355071736704529973127"),
    "peer record D,M^8,dual P,M,M,T(3,5)": (["D"] + ["M"] * 8 + ["dP", "M", "M", "T35"], "4.506428462939128775"),
    "peer D,M^12": (["D"] + ["M"] * 12, "4.506428305525435"),
    "peer beam M,D^8,P,D,D,dual T(4,3),T(3,5),T(3,5)": (["M"] + ["D"] * 8 + ["P", "D", "D", "dT43", "T35", "T35"],
                                                         "4.506428465445233177909286"),
}
if len(sys.argv) > 2:  # record.py <prec> <name substring>: only the matching words
    WORDS = {k: v for k, v in WORDS.items() if v and sys.argv[2] in k}
    sys.argv = sys.argv[:2]

if __name__ == "__main__":
    for prec in sys.argv[1:] or ["double", "mp"]:
        A = Arith(prec)
        for name, item in WORDS.items():
            if item is None:
                continue
            word, claim = item
            r, st = tower(word, A)
            diff = mp.mpf(str(r)) - mp.mpf(claim)
            print(f"[{prec}] {name}: {mp.nstr(r, 30) if prec == 'mp' else repr(r)}   claimed {claim}   diff {mp.nstr(diff, 3)}")
