"""Check of Proposition 5.1: F(x,y) = [|y|>=m+1] OR (x AND y_A), |A| = m, k = 2m+1. Brute-force E_x/Inf_x and
the influences, compared with hidden.json."""
import json, math, sys
import numpy as np
from composition import brute_E
for m in range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(1, 6):
    k = 2 * m + 1; n = k + 1
    xs = np.arange(2 ** n)
    x = xs & 1
    y = [(xs >> (j + 1)) & 1 for j in range(k)]
    wt = sum(y)
    yA = np.ones_like(xs)
    for j in range(m):
        yA &= y[j]
    val = (wt >= m + 1) | (x & yA).astype(bool)
    tt = np.where(val, -1.0, 1.0)
    E, d = brute_E(tt, n, 0)
    infs = [brute_E(tt, n, i)[1] for i in range(n)]
    print(f"k={k:2d} Inf_x={d:.3e} (2^-k={2.0**-k:.3e}) E_x/Inf_x={E/d:.6f}  max other Inf={max(infs[1:]):.3e}")
