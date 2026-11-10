"""Which sign convention for M and D reproduces the claimed tower values?  Each letter may stand for any of the four
variants g(x), -g(x), g(-x), -g(-x) of PM(K33); words are read outermost-first or innermost-first."""
import itertools
import numpy as np
from composition import step, E_of, DICTATOR
from towers import M, K

FLIP = 2 ** K - 1
VAR = {"id": M, "neg_out": -M, "neg_in": M[[FLIP ^ y for y in range(2 ** K)]],
       "dual": -M[[FLIP ^ y for y in range(2 ** K)]]}
CLAIMS = {"MDDMMDD": 4.50504996749762, "MDDDDDMM": 4.506379879463887}


def run(word, gm, gd, outer_first):
    st = DICTATOR
    seq = reversed(word) if outer_first else iter(word)
    for ch in seq:
        st = step(st, gm if ch == "M" else gd, K, 0)
    E, Inf = E_of(st)
    return E / Inf, st[0]


for (nm, gm), (nd, gd) in itertools.product(VAR.items(), VAR.items()):
    for of in (True, False):
        vals = {w: run(w, gm, gd, of) for w in CLAIMS}
        hit = all(abs(vals[w][0] - c) < 1e-9 for w, c in CLAIMS.items())
        line = "  ".join(f"{w}={vals[w][0]:.12f} (mean {vals[w][1]:+.4f})" for w in CLAIMS)
        print(f"M={nm:8s} D={nd:8s} {'outer-first' if of else 'inner-first'}  {line}  {'MATCH' if hit else ''}")
