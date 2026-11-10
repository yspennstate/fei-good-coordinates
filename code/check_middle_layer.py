"""Check of Proposition 5.2: F(y,z) = [|y|>=m+1] OR OR_{|A|=m} (z_A AND y_A), k = 2m+1.
(a) brute force at k=3,5 (n = k + C(k,m) variables): hidden and base E/Inf, influences;
(b) formula (5.1) G_hidden = 2^-k sum_T Phi((2^k ahat(T))^2, 1), a = +1 below, 0 on, -1 above the middle layer;
(c) base identity E_base/Inf_base = 1 + E_base(a)/(r delta), E_base(a) = erasure sum of a's coefficients at a base bit;
(d) (5.2) ahat(T) = (1 - s/(k+1)) Majhat(T) at odd s."""
import itertools, math
import numpy as np
from composition import brute_E, walsh, m as mfun

def Phi(a, b):
    return float(mfun(np.array([a]), np.array([b]))[0])

def a_coeffs(k):
    ys = np.arange(2 ** k)
    wt = np.array([bin(v).count("1") for v in ys])
    m = (k - 1) // 2
    a = np.where(wt <= m - 1, 1.0, np.where(wt == m, 0.0, -1.0))
    maj = np.where(wt <= m, 1.0, -1.0)
    return walsh(a), walsh(maj), wt

def formula(k):
    ah, mh, wt = a_coeffs(k)
    G = 2.0 ** -k * sum(Phi((2 ** k * c) ** 2, 1) for c in ah)
    s = np.array([bin(T).count("1") for T in range(2 ** k)])
    odd = s % 2 == 1
    err52 = np.max(np.abs(ah[odd] - (1 - s[odd] / (k + 1)) * mh[odd]))
    m = (k - 1) // 2
    r, delta = math.comb(k, m), 2.0 ** -k
    A = np.array([T for T in range(2 ** k) if not T & 1])
    Ea = float(mfun(ah[A] ** 2, ah[A | 1] ** 2).sum())
    return G, 1 + Ea / (r * delta), err52

for k in (3, 5):
    m = (k - 1) // 2
    subsets = list(itertools.combinations(range(k), m))
    n = k + len(subsets)
    xs = np.arange(2 ** n)
    y = [(xs >> j) & 1 for j in range(k)]
    wt = sum(y)
    val = wt >= m + 1
    for h, A in enumerate(subsets):
        t = (xs >> (k + h)) & 1
        for j in A:
            t = t & y[j]
        val = val | t.astype(bool)
    tt = np.where(val, -1.0, 1.0)
    Eb, db = brute_E(tt, n, 0)
    Eh, dh = brute_E(tt, n, k)
    G, base, err52 = formula(k)
    print(f"k={k} n={n}: brute hidden {Eh/dh:.12f} (Inf {dh:.3e}={2.0**-k:.3e}) vs (5.1) {G:.12f}; "
          f"brute base {Eb/db:.12f} (Inf {db:.4e} = r delta {len(subsets)*2.0**-k:.4e}) vs identity {base:.12f}; (5.2) err {err52:.1e}")
for k in range(7, 18, 2):
    G, base, err52 = formula(k)
    print(f"k={k:2d} (5.1) hidden {G:.12f}  base {base:.12f}  (5.2) err {err52:.1e}")
