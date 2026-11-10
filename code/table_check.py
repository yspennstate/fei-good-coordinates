"""Table 1 of the paper: every tower at 40 and at 60 digits, the variable count of each word, and the small witness."""
import math
import mpmath as mp
import record as R

R.GATES["dT25"] = (R.dual(R.tribes(2, 5)), 10)
WORDS = {
    "M D D D D dT25": ["M", "D", "D", "D", "D", "dT25"],
    "M D D M M D D": list("MDDMMDD"),
    "M D D D D D M M": list("MDDDDDMM"),
    "D M^12": ["D"] + ["M"] * 12,
    "B14": ["M", "D", "D", "D", "D", "M", "P", "P", "P", "Q2", "dPMK4", "dT43", "T26", "T35"],
    "D M^8 dP M M T35": ["D"] + ["M"] * 8 + ["dP", "M", "M", "T35"],
    "M D^8 P D D dT43 T35 T35": ["M"] + ["D"] * 8 + ["P", "D", "D", "dT43", "T35", "T35"],
}
for name, w in WORDS.items():
    nvar = math.prod(R.GATES[g][1] for g in w)
    vals = []
    for dps in (40, 60):
        mp.mp.dps = dps
        r, _ = R.tower(w, R.Arith("mp"))
        vals.append(r)
    mp.mp.dps = 60
    print(f"{name}: n={nvar} ({nvar:.2e})  40d={mp.nstr(vals[0], 30)}  60d={mp.nstr(vals[1], 30)}  diff={mp.nstr(vals[1]-vals[0], 3)}",
          flush=True)
