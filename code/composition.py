"""E_i and Inf_i of a coordinate under block composition f = g(h, ..., h).

Spectral sample U ~ fhat^2.  E_i(f) = H(1[i in U] | U minus i) in bits.
For a leaf i inside block j the state carried up is
  (mu, P1, e, Inf):  mu = E f,  P1 = fhat({i})^2,
  e = the part of E_i over sets A = U minus i that are nonempty,  Inf = Inf_i(f).
With ghat the mu-biased coefficients of the gate (inputs of mean mu, sigma^2 = 1 - mu^2),
  W_j  = sum_{T containing j} ghat(T)^2
  mu'  = ghat(empty)
  P1'  = ghat({j})^2 P1 / sigma^2
  e'   = (W_j / sigma^2) e + sum_{T not containing j, T nonempty} m(ghat(T)^2, ghat(T+j)^2 P1 / sigma^2)
  Inf' = W_j Inf / sigma^2
  E'   = e' + m(mu'^2, P1'),     m(a, b) = (a + b) h2(b / (a + b)).
"""
import itertools
import numpy as np


def m(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = np.where(a > 0, a * np.log2(np.where(a > 0, s / np.where(a > 0, a, 1), 1)), 0.0)
        tb = np.where(b > 0, b * np.log2(np.where(b > 0, s / np.where(b > 0, b, 1), 1)), 0.0)
    return ta + tb


def biased_coeffs(tt, k, mu):
    """tt: array of length 2^k, tt[y] = g(y) in {+1,-1}; bit j of y set means y_j = -1.
    Returns c[T] (T as bitmask) = E[g(y) prod_{j in T} (y_j - mu)/sigma], y_j iid mean mu."""
    sig = np.sqrt(1 - mu * mu)
    ys = np.array([[1 - 2 * ((y >> j) & 1) for j in range(k)] for y in range(2 ** k)], float)
    py = np.prod(np.where(ys > 0, (1 + mu) / 2, (1 - mu) / 2), axis=1)
    phi = (ys - mu) / sig
    c = np.zeros(2 ** k)
    for T in range(2 ** k):
        prod = np.ones(2 ** k)
        for j in range(k):
            if (T >> j) & 1:
                prod = prod * phi[:, j]
        c[T] = np.sum(py * tt * prod)
    return c


def step(state, tt, k, j):
    mu, P1, e, Inf = state
    s2 = 1 - mu * mu
    c = biased_coeffs(tt, k, mu)
    c2 = c * c
    bit = 1 << j
    T = np.arange(2 ** k)
    has = (T & bit) != 0
    W = c2[has].sum()
    rest = T[(~has) & (T != 0)]
    e_new = W / s2 * e + m(c2[rest], c2[rest | bit] * P1 / s2).sum()
    P1_new = c2[bit] * P1 / s2
    return (c[0], P1_new, e_new, W * Inf / s2)


def E_of(state):
    mu, P1, e, Inf = state
    return float(e + m(mu * mu, P1)), Inf


DICTATOR = (0.0, 1.0, 0.0, 1.0)


# ---------- brute force on an explicit truth table ----------
def walsh(tt):
    a = np.array(tt, float).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            x = a[i:i + h].copy(); y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y; a[i + h:i + 2 * h] = x - y
        h *= 2
    return a / len(a)


def brute_E(tt, n, i):
    f2 = walsh(tt) ** 2
    bit = 1 << i
    A = np.array([s for s in range(2 ** n) if not s & bit])
    return float(m(f2[A], f2[A | bit]).sum()), float(f2[A | bit].sum())


def compose_tt(g, k, h, r):
    """f(x) = g(h(x_block0), ..., h(x_block_{k-1})), blocks of r bits, block j = bits j*r..j*r+r-1."""
    n = k * r
    out = np.empty(2 ** n)
    for x in range(2 ** n):
        y = 0
        for j in range(k):
            if h[(x >> (j * r)) & ((1 << r) - 1)] < 0:
                y |= 1 << j
        out[x] = g[y]
    return out
