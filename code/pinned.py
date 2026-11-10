"""Claim: M^k with the input mean held at PM(K33)'s fixed point tends to 4.5064554516.
Find the fixed point mu* of the mean map mu -> c_0(mu) of M, then iterate step() with M from states (mu*, P1, e, Inf)
with different leaf data; the ratio limit should not depend on them."""
import mpmath as mp
from record import Arith, GATES, tt_from, step, biased_coeffs

A = Arith("mp")
fn, k = GATES["M"]
tt = tt_from(fn, k)
g = lambda mu: biased_coeffs(tt, k, mu, A)[0] - mu
lo, hi = mp.mpf("-0.2"), mp.mpf("0.1")
assert g(lo) * g(hi) < 0
for _ in range(140):
    mid = (lo + hi) / 2
    if g(lo) * g(mid) <= 0: hi = mid
    else: lo = mid
mu = (lo + hi) / 2
print("mu* =", mp.nstr(mu, 25), " slope of the mean map there:", mp.nstr(mp.diff(lambda x: biased_coeffs(tt, k, x, A)[0], mu), 12))
for P1, e, Inf in [(1 - mu * mu, 0, 1 - mu * mu), (mp.mpf("0.3"), mp.mpf("0.5"), mp.mpf("0.4")), (mp.mpf("0.01"), mp.mpf("2"), mp.mpf("0.2"))]:
    st = (mu, P1, e, Inf)
    vals = []
    for it in range(40):
        st = (mu,) + step(st, tt, k, A)[1:]   # hold the mean exactly at mu*
        m_, P, ee, I = st
        vals.append((ee + A.m(m_ * m_, P)) / I)
    print("start", [mp.nstr(v, 4) for v in (P1, e, Inf)], "->", " ".join(mp.nstr(vals[i], 14) for i in (4, 9, 19, 39)))

print("-- realizable starts: the actual state after a bottom word, mean replaced by mu*, then M^k pinned --")
D = tt_from(*GATES["D"]), GATES["D"][1]
for bottom in (["D"], ["D", "M"], ["D", "M", "M"], ["M"]):
    st = (A.f(0), A.f(1), A.f(0), A.f(1))
    for gname in bottom:
        t, kk = tt_from(*GATES[gname]), GATES[gname][1]
        st = step(st, t, kk, A)
    m0, P, ee, I = st
    print(f"  {','.join(bottom)}: actual mean {mp.nstr(m0, 8)}, P1/Inf^2 = {mp.nstr(P / I ** 2, 8)}")
    st = (mu,) + st[1:]
    for it in range(60):
        st = (mu,) + step(st, tt, k, A)[1:]
    m_, P, ee, I = st
    print("     pinned M^60 limit:", mp.nstr((ee + A.m(m_ * m_, P)) / I, 14))
