"""Check of Theorem 4.2: random width-w DNFs with negated literals.
Checks (a) inequality (2.1) at every coordinate, (b) the term-coordinate claim: in any term, the
literal coordinate of largest influence has E_i/Inf_i <= 2 + log2 e + log2(2^(k+1)-1), k = term size."""
import math, random, sys
import numpy as np
from composition import brute_E

LOG2E = 1 / math.log(2)
rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
worst21, worst_term, count = -1e9, -1e9, 0
for trial in range(3000):
    n = rng.randint(3, 10)
    w = rng.randint(1, min(4, n))
    terms = []
    for _ in range(rng.randint(1, 6)):
        k = rng.randint(1, w)
        vs = rng.sample(range(n), k)
        terms.append([(v, rng.random() < 0.5) for v in vs])
    xs = np.arange(2 ** n)
    bits = [(xs >> v) & 1 for v in range(n)]
    val = np.zeros(2 ** n, bool)
    for t in terms:
        sat = np.ones(2 ** n, bool)
        for v, neg in t:
            sat &= (bits[v] == 0) if neg else (bits[v] == 1)
        val |= sat
    if val.all() or not val.any():
        continue
    tt = np.where(val, -1.0, 1.0)
    V = 1 - tt.mean() ** 2
    EI = [brute_E(tt, n, i) for i in range(n)]
    for i, (E, d) in enumerate(EI):
        if d > 1e-12:
            worst21 = max(worst21, E / d - (2 + LOG2E + math.log2(1 + V / d)))
    for t in terms:
        k = len(t)
        j = max((v for v, _ in t), key=lambda v: EI[v][1])
        E, d = EI[j]
        if d > 1e-12:
            worst_term = max(worst_term, E / d - (2 + LOG2E + math.log2(2 ** (k + 1) - 1)))
        else:
            worst_term = max(worst_term, 1e9)
    count += 1
print(f"{count} nonconstant DNFs; max over coords of E/Inf - rhs(2.1) = {worst21:.4f} (must be <= 0); "
      f"max over terms of E/Inf(max-inf literal) - bound = {worst_term:.4f} (must be <= 0)")
