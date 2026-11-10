"""Check of Theorem 4.3 (every leaf of a read-once AND/OR formula has E_i/Inf_i < pi/ln 2).
(a) formula (3.3) against brute force on chains f_l = c_l + (f_{l-1}-c_l) Z_l with arbitrary guard functions Z_l;
(b) integral_1^inf F(u)/u^2 du = pi/ln 2, F(u) = Phi(u^2,1);
(c) the potential inequalities -V_alpha >= f(beta)/s^2 and V >= f(alpha)/s on a grid, alpha >= beta >= 0;
(d) the sharpness family (alternating gates, pivotal q = 1-2^-m, K = m 2^m gates) through (3.3)."""
import math, random
import numpy as np
import mpmath as mp
from composition import brute_E

L2 = math.log(2)
def Phi(a, b):
    s = a + b
    return (a * math.log2(s / a) if a > 0 else 0) + (b * math.log2(s / b) if b > 0 else 0)

def formula(ps, cs):
    P = [1.0]
    for p in ps:
        P.append(P[-1] * p)
    delta = P[-1]
    out, acc = 0.0, 0.0
    for r in range(1, len(ps) + 1):
        R = acc - cs[r - 1] / P[r - 1]
        out += (P[r - 1] - P[r]) * Phi(R * R, 1)
        acc += cs[r - 1] * (1 - ps[r - 1]) / P[r]
    out += delta * Phi(acc * acc, 1)
    return out

rng = random.Random(7)
worst = 0.0
for trial in range(300):
    L = rng.randint(1, 4)
    sizes = [rng.randint(1, 3) for _ in range(L)]
    if 1 + sum(sizes) > 12:
        continue
    n = 1 + sum(sizes)
    xs = np.arange(2 ** n)
    f = np.where(xs & 1, -1.0, 1.0)
    ps, cs, off = [], [], 1
    for r in sizes:
        while True:
            Zt = np.array([rng.random() < 0.5 for _ in range(2 ** r)], float)
            if 0 < Zt.mean() < 1:
                break
        Z = Zt[(xs >> off) & (2 ** r - 1)]
        c = rng.choice([-1.0, 1.0])
        f = c + (f - c) * Z
        ps.append(Zt.mean()); cs.append(c); off += r
    E, d = brute_E(f, n, 0)
    worst = max(worst, abs(E / d - formula(ps, cs)))
print(f"(a) 3.3 vs brute force: max |diff| = {worst:.2e}")

mp.mp.dps = 30
F = lambda u: u * u * mp.log(1 + 1 / (u * u), 2) + mp.log(1 + u * u, 2)
I = mp.quad(lambda u: F(u) / u ** 2, [1, 10, 100, mp.inf])
print(f"(b) integral = {mp.nstr(I, 20)}, pi/ln2 = {mp.nstr(mp.pi / mp.log(2), 20)}")

Ff = lambda u: Phi(u * u, 1)
fs = lambda t: Ff(2 * t + 1)
def fprime(t, h=1e-6):
    return (fs(t + h) - fs(t - h)) / (2 * h) if t > h else (fs(t + h) - fs(t)) / h
def W(a):
    return float(mp.quad(lambda t: 2 * fs(float(t)) / (1 + 2 * t) ** 2, [a, a + 10, mp.inf]))
def V(a, b):
    s = 1 + a + b
    return fs(a) * (1 / s - 1 / (1 + 2 * a)) + W(a)
worst_c, worst_v = 1e9, 1e9
for a in [0, 0.01, 0.1, 0.3, 1, 2, 5, 20, 100]:
    for frac in [0, 0.1, 0.5, 0.9, 1.0]:
        b = a * frac
        s = 1 + a + b
        h = 1e-5
        Va = (V(a + h, b) - V(a - h, b)) / (2 * h) if a > h else (V(a + h, b) - V(a, b)) / h
        worst_c = min(worst_c, -Va - fs(b) / s ** 2)
        worst_v = min(worst_v, V(a, b) - fs(a) / s)
print(f"(c) min(-V_alpha - f(beta)/s^2) = {worst_c:.3e} (>= -1e-6 expected); min(V - f(alpha)/s) = {worst_v:.3e} (> 0)")
print(f"    V(0,0) = {V(0, 0):.12f}")

for m in range(2, 15, 2):
    q, K = 1 - 2.0 ** -m, m * 2 ** m
    cs = [(-1.0) ** l for l in range(K)]
    print(f"(d) m={m:2d} K={K:6d} E/Inf = {formula([q] * K, cs):.6f}")
