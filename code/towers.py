"""E_i / Inf_i along towers of PM(K33) and its dual, with the recursion of composition.py.

M(x) = +1 when the edges of K_{3,3} carrying +1 contain a perfect matching, else -1 (9 inputs, edge (a, b) is bit 3a + b).
D(x) = -M(-x), the dual. Both gates are invariant under the automorphisms of K_{3,3}, which act transitively on edges,
so every leaf of a tower has the same ratio and the leaf position inside each gate does not matter.
A word w_1 w_2 ... w_d means f = w_1(w_2(...w_d(x)...)): w_d is the gate next to the variables.
"""
import itertools
import sys
import numpy as np
from composition import step, E_of, DICTATOR

K = 9
PERMS = list(itertools.permutations(range(3)))


def pm_k33():
    tt = np.empty(2 ** K)
    for y in range(2 ** K):
        plus = [((y >> e) & 1) == 0 for e in range(K)]  # bit clear = +1
        tt[y] = 1.0 if any(all(plus[3 * a + p[a]] for a in range(3)) for p in PERMS) else -1.0
    return tt


M = pm_k33()
D = np.array([-M[(2 ** K - 1) ^ y] for y in range(2 ** K)])  # D(x) = -M(-x)
GATES = {"M": M, "D": D}


def ratio(word):
    st = DICTATOR
    for g in reversed(word):          # innermost gate first
        st = step(st, GATES[g], K, 0)
    E, Inf = E_of(st)
    return E / Inf, st


if __name__ == "__main__":
    assert abs(M.mean()) < 1 and abs(D.mean()) < 1
    print(f"E M = {M.mean():.6f}, E D = {D.mean():.6f}  (M has 2^9 * P(PM) +1 outputs)")
    for w in sys.argv[1:] or ["M", "D", "MD", "DM", "MDDMMDD", "MDDDDDMM", "DMMMMMDD"]:
        r, st = ratio(w)
        print(f"{w:12s} depth {len(w)}  E/Inf = {r:.15f}   mean {st[0]:+.6f}")
