"""Recompute of the tower ratios E_i/Inf_i (bits per unit influence).

E_i = H(1[i in U] | U minus i), U drawn from the squared Fourier spectrum. For a composition f = g(h(x^1),...,h(x^m))
with disjoint blocks, U = union of U_k where T ~ g's mu-biased squared spectrum (mu = E h) and, for k in T, U_k is
drawn from h's spectrum conditioned nonempty. A leaf is tracked by (mean, z=P(U empty), a=P(U={i}), e, inf) where
e is the part of E_i from nonempty U minus i. Checked against brute force.
"""
import itertools
import math
import sys

import numpy as np


def H2(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def table(fn, m):
    """+-1 truth table of a 0/1 function on m bits; index bit k = input k (1 = true). Output +1 = false, -1 = true."""
    t = np.empty(1 << m)
    for x in range(1 << m):
        t[x] = -1.0 if fn([(x >> k) & 1 for k in range(m)]) else 1.0
    return t


def biased_coeffs(t, m, mu):
    """g~(T) for inputs y in {+-1} with E y = mu (input true <-> y = -1, matching the table's output convention)."""
    # input bit 1 means the lower function is TRUE, i.e. its +-1 value is -1
    p_true = (1 - mu) / 2  # P(y = -1)
    s = math.sqrt(max(1 - mu * mu, 0.0))
    phi_t = (-1 - mu) / s  # phi at y = -1 (bit 1)
    phi_f = (1 - mu) / s   # phi at y = +1 (bit 0)
    c = t.copy()
    for k in range(m):
        c = c.reshape(-1, 2, 1 << k)
        v0, v1 = c[:, 0, :].copy(), c[:, 1, :].copy()  # bit k = 0 (false), 1 (true)
        c[:, 0, :] = (1 - p_true) * v0 + p_true * v1
        c[:, 1, :] = (1 - p_true) * v0 * phi_f + p_true * v1 * phi_t
        c = c.reshape(-1)
    return c  # index bit k set <-> k in T


def compose(state, t, m, j=0):
    mean, z, a, e, inf = state
    c2 = biased_coeffs(t, m, mean) ** 2
    assert abs(c2.sum() - 1) < 1e-9, c2.sum()
    bj = 1 << j
    Pj = c2[[x for x in range(1 << m) if x & bj]].sum()
    r = 1.0 / (1 - z)
    Ei = Pj * e * r
    for S in range(1 << m):
        if S & bj:
            continue
        w0, w1 = c2[S], c2[S | bj] * a * r
        if w0 + w1 > 0:
            Ei += (w0 + w1) * H2(w1 / (w0 + w1))
    newmean = biased_coeffs(t, m, mean)[0]
    z2 = newmean ** 2
    a2 = c2[bj] * a * r
    inf2 = Pj * inf * r
    e2 = Ei - ((z2 + a2) * H2(a2 / (z2 + a2)) if z2 + a2 > 0 else 0.0)
    return (newmean, z2, a2, e2, inf2), Ei / inf2


LEAF = (0.0, 0.0, 1.0, 0.0, 1.0)


def run(word):
    st, ratio = LEAF, None
    for t, m in word:
        st, ratio = compose(st, t, m)
    return ratio, st


# ---- gates (0/1 functions; input k is bit k) ----
def T(w, s):
    return lambda x: any(all(x[b * w + i] for i in range(w)) for b in range(s)), w * s


def dual(g):
    fn, m = g
    return (lambda x: not fn([1 - v for v in x])), m


def PM(n):  # perfect matching on n x n bipartite graph, edge (r,c) = bit r*n+c
    perms = list(itertools.permutations(range(n)))
    return (lambda x: any(all(x[r * n + p[r]] for r in range(n)) for p in perms)), n * n


def PMK4():  # three pairs of opposite edges of K4 = three disjoint pairs
    return T(2, 3)


def G(g):
    fn, m = g
    return table(fn, m), m


def brute_ratio(fn, n):
    """min over coordinates of E_i/Inf_i by full Walsh transform."""
    t = table(fn, n)
    c = t.copy()
    for k in range(n):
        c = c.reshape(-1, 2, 1 << k)
        v0, v1 = c[:, 0, :].copy(), c[:, 1, :].copy()
        c[:, 0, :], c[:, 1, :] = (v0 + v1) / 2, (v0 - v1) / 2
        c = c.reshape(-1)
    p = c ** 2
    out = []
    for i in range(n):
        bi = 1 << i
        Ei = sum((p[S] + p[S | bi]) * H2(p[S | bi] / (p[S] + p[S | bi]))
                 for S in range(1 << n) if not S & bi and p[S] + p[S | bi] > 0)
        Inf = sum(p[S] for S in range(1 << n) if S & bi)
        out.append(Ei / Inf)
    return min(out), max(out)


def composed_fn(gates):
    """0/1 function of a bottom-to-top word, for brute-force checks."""
    def build(level):
        fn, m = gates[level]
        if level == 0:
            return fn, m
        lo, lm = build(level - 1)
        return (lambda x: fn([lo(x[k * lm:(k + 1) * lm]) for k in range(m)])), m * lm
    return build(len(gates) - 1)


def main():
    # brute-force checks of the recursion
    checks = [[T(2, 2), T(2, 2)], [T(2, 2), dual(T(2, 2))], [dual(T(2, 2)), T(2, 3)], [T(2, 3), T(2, 2)],
              [PM(2), dual(T(2, 2))], [T(3, 1), T(2, 2)], [T(2, 2), T(2, 2), T(2, 1)]]
    for gs in checks:
        fn, n = composed_fn(gs)
        if n > 18:
            continue
        rec, _ = run([G(g) for g in gs])
        lo, hi = brute_ratio(fn, n)
        print(f"check n={n:2d} recursion {rec:.12f} brute min {lo:.12f} max {hi:.12f} diff {abs(rec-lo):.1e}",
              flush=True)

    M, P, D = PM(3), PM(4), dual(PM(3))
    words = {
        "old tower M,D,D,M,M,D,D": [M, D, D, M, M, D, D],
        "small M,D,D,D,D,dual(T(2,5))": [M, D, D, D, D, dual(T(2, 5))],
        "record M,D,D,D,D,M,P,P,P,Q2,dual(PM(K4)),dual(T(4,3)),T(2,6),T(3,5)":
            [M, D, D, D, D, M, P, P, P, dual(T(2, 2)), dual(PMK4()), dual(T(4, 3)), T(2, 6), T(3, 5)],
    }
    cache = {}
    for name, gs in words.items():
        word = []
        for g in gs:
            key = id(g)
            if key not in cache:
                cache[key] = G(g)
            word.append(cache[key])
        ratio, st = run(word)
        nvars = 1
        for _, m in word:
            nvars *= m
        print(f"{name}: ratio {ratio:.14f} vars {nvars:,} mean {st[0]:+.6f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
