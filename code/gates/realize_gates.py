"""Can the higher pinned limits (Q2 at mean 0.236068 -> 4.5168, dual T(4,3) at 0.610319 -> 4.5107) be realised?
Search bottom words whose output mean lands near the gate's fixed point, then stack the gate unpinned."""
import json
from tower_recompute import PM, T, dual, G, compose, LEAF
gates = {"M": G(PM(3)), "D": G(dual(PM(3))), "P": G(PM(4)), "Pd": G(dual(PM(4))), "Q2": G(dual(T(2, 2))),
         "T26": G(T(2, 6)), "dT26": G(dual(T(2, 6))), "T35": G(T(3, 5)), "dT35": G(dual(T(3, 5))), "dT43": G(dual(T(4, 3)))}
targets = {"Q2": 0.2360679775, "dT43": 0.610319}
def step(st, g):
    try:
        return compose(st, *gates[g])
    except AssertionError:
        return None
layer = {(): LEAF}
allw = {}
for d in range(1, 5):
    nl = {}
    for w, st in layer.items():
        for g in gates:
            r = step(st, g)
            if r:
                nl[w + (g,)] = r[0]
    layer = nl
    allw.update(nl)
    print("depth", d, len(nl), flush=True)
for tn, tm in targets.items():
    near = sorted(layer, key=lambda w: abs(layer[w][0] - tm))[:300]
    for w in near:
        for g in gates:
            r = step(layer[w], g)
            if r:
                allw[w + (g,)] = r[0]
out = {}
for tn, tm in targets.items():
    cand = sorted(allw, key=lambda w: abs(allw[w][0] - tm))[:400]
    best = []
    for w in cand:
        st, rs = allw[w], []
        for k in range(40):
            r = step(st, tn)
            if not r:
                break
            st = r[0]
            rs.append(r[1])
        if rs:
            k = rs.index(max(rs))
            best.append((max(rs), ",".join(w), k + 1, abs(allw[w][0] - tm)))
    best.sort(reverse=True)
    out[tn] = best[:10]
    print(tn, "closest dev", min(abs(allw[w][0] - tm) for w in cand))
    for b in best[:10]:
        print(f"  {b[0]:.10f}  {b[1]} + {tn}^{b[2]}  dev {b[3]:.2e}", flush=True)
json.dump(out, open("realize_gates.json", "w"), indent=1)
