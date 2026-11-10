"""Interval certificate for the monotone FEI tower records (rigorous bounds).

Same composition recursion as tower_recompute.py, every number an mpmath interval (outward rounding),
so each printed ratio interval contains the exact value of the recursion. Truth tables are exact +-1 integers and the
start state is exact. Prints, per word, the enclosure of E/Inf at the top level and whether its LOWER endpoint beats
B14's 4.506400645515568.
"""
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tower_recompute import PM, T, dual, PMK4, table  # noqa: E402

iv = mp.iv
iv.dps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
LN2 = iv.log(2)
ZERO, ONE = iv.mpf(0), iv.mpf(1)
B14 = mp.mpf("4.506400645515568")


def is_zero(x):
    return x.a == 0 and x.b == 0


def H2(p):
    # binary entropy; caller has excluded exact 0 and 1. p must lie strictly inside (0, 1).
    assert p.a > 0 and p.b < 1, ("H2 argument not inside (0,1)", p)
    return -(p * iv.log(p) + (1 - p) * iv.log(1 - p)) / LN2


def wH2(w0, w1):
    """(w0+w1) * H2(w1/(w0+w1)), with the exact-zero cases giving 0."""
    if is_zero(w1) or is_zero(w0):
        return ZERO
    s = w0 + w1
    return s * H2(w1 / s)


def biased_c2(t, m, mu):
    p_true = (1 - mu) / 2
    q_f = 1 - p_true
    s = iv.sqrt(1 - mu * mu)
    phi_t, phi_f = (-1 - mu) / s, (1 - mu) / s
    c = np.array([iv.mpf(int(v)) for v in t], dtype=object)
    for k in range(m):
        c = c.reshape(-1, 2, 1 << k)
        v0, v1 = c[:, 0, :].copy(), c[:, 1, :].copy()
        # array on the left: an iv scalar on the left refuses a numpy operand
        c[:, 0, :] = v0 * q_f + v1 * p_true
        c[:, 1, :] = v0 * (q_f * phi_f) + v1 * (p_true * phi_t)
        c = c.reshape(-1)
    return c


def compose(state, t, m):
    mean, z, a, e, inf = state
    c = biased_c2(t, m, mean)
    c2 = c * c
    r = 1 / (1 - z)
    Pj = sum(c2[1::2], ZERO)
    Ei = Pj * e * r
    for S in range(0, 1 << m, 2):
        Ei += wH2(c2[S], c2[S + 1] * a * r)
    newmean = c[0]
    z2, a2, inf2 = newmean ** 2, c2[1] * a * r, Pj * inf * r
    e2 = Ei - wH2(z2, a2)
    return (newmean, z2, a2, e2, inf2), Ei / inf2


def run(word, gates):
    st = (ZERO, ZERO, ONE, ZERO, ONE)
    tabs = {}
    for name in word:
        if name not in tabs:
            g = gates[name]
            tabs[name] = (table(*g), g[1])
        st, ratio = compose(st, *tabs[name])
    return ratio, st[0]


def main():
    P4 = PM(4)
    gates = {"M": PM(3), "D": dual(PM(3)), "P": PM(4), "Pd": dual(P4), "dK4": dual(PMK4()), "dT43": dual(T(4, 3)),
             "T35": T(3, 5)}
    words = {
        "D,M^12": ["D"] + ["M"] * 12,
        "D,M^8,dual PM(4),M,M,T(3,5)": ["D"] + ["M"] * 8 + ["Pd", "M", "M", "T35"],
        "M,D^8,P,D,D,dual T(4,3),T(3,5),T(3,5)": ["M"] + ["D"] * 8 + ["P", "D", "D", "dT43", "T35", "T35"],
    }
    print(f"interval dps {iv.dps}; B14 = {B14}", flush=True)
    pick = {int(i) for i in sys.argv[2].split(",")} if len(sys.argv) > 2 else None
    for k, (label, w) in enumerate(words.items()):
        if pick is not None and k not in pick:
            continue
        ratio, mean = run(w, gates)
        lo, hi = ratio.a, ratio.b
        width = hi - lo
        beats = lo > B14
        print(f"{label}\n  E/Inf in [{mp.nstr(lo, 25)}, {mp.nstr(hi, 25)}]  width {mp.nstr(width, 3)}"
              f"  top mean {mp.nstr(mean.mid, 10)}  lower end > B14: {beats}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
