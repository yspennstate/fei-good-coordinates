"""Brute-force check of the composition law in composition.py on 9- to 16-variable compositions."""
import numpy as np
from composition import step, E_of, DICTATOR, brute_E, compose_tt

rng = np.random.default_rng(20261110)
worst = 0.0
cases = 0
for trial in range(60):
    k = int(rng.integers(2, 5)); r = int(rng.integers(2, 5))
    if k * r > 16:
        continue
    pg = rng.uniform(0.15, 0.85); ph = rng.uniform(0.15, 0.85)
    g = np.where(rng.random(2 ** k) < pg, -1.0, 1.0)
    h = np.where(rng.random(2 ** r) < ph, -1.0, 1.0)
    if abs(g.sum()) == len(g) or abs(h.sum()) == len(h):
        continue  # constant gate or leaf: sigma = 0, nothing to check
    f = compose_tt(g, k, h, r)
    n = k * r
    for j in range(k):
        for t in range(r):
            i = j * r + t
            Eb, Ib = brute_E(f, n, i)
            # leaf state of coordinate t inside h, then one composition step at position j
            st = step(DICTATOR, h, r, t)
            st = step(st, g, k, j)
            Er, Ir = E_of(st)
            assert np.isfinite([Eb, Er, Ib, Ir]).all(), (k, r, j, t)
            worst = max(worst, abs(Eb - Er), abs(Ib - Ir))
            cases += 1
print(f"{cases} coordinates in random compositions g(h,...,h) of 4 to 16 variables: max |E or Inf difference| = {worst:.2e}")
