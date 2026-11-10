"""Section 5 checks: (i) the uniform lower bound |2^k Maj^(T)| >= 2^(m+1)/((2m+1)(m+1)) on odd T;
(ii) L^(T) = s Maj^(T)/(k+1) on odd T (L = middle-layer indicator); (iii) the spectral sample of x1 AND (x2 OR x3)."""
import itertools, math
from fractions import Fraction as Fr

def wht(vals, k):
    c = [Fr(v) for v in vals]
    h = 1
    while h < len(c):
        for i in range(0, len(c), 2 * h):
            for t in range(i, i + h):
                a, b = c[t], c[t + h]
                c[t], c[t + h] = (a + b) / 2, (a - b) / 2
        h *= 2
    return c

for k in (3, 5, 7, 9, 11, 13):
    m = (k - 1) // 2
    maj = [1 if bin(y).count("1") <= m else -1 for y in range(2 ** k)]
    L = [1 if bin(y).count("1") == m else 0 for y in range(2 ** k)]
    M, Lh = wht(maj, k), wht(L, k)
    odd = [T for T in range(2 ** k) if bin(T).count("1") % 2]
    mn = min(abs(2 ** k * M[T]) for T in odd)
    bound = Fr(2 ** (m + 1), (2 * m + 1) * (m + 1))
    ok = all(Lh[T] == Fr(bin(T).count("1"), k + 1) * M[T] for T in odd)
    print(f"k={k}: min odd |2^k Maj^| = {float(mn):.4f} >= bound {float(bound):.4f}: {mn >= bound};  L^ = s Maj^/(k+1): {ok}")

# x1 AND (x2 OR x3), true = -1
f = [(-1 if ((y & 1) and ((y >> 1) & 1 or (y >> 2) & 1)) else 1) for y in range(8)]
c2 = [v * v for v in wht(f, 3)]
P = lambda S: sum(c2[T] for T in range(8) if T & S == S)
print("sq coeffs x16:", [int(v * 16) for v in c2], " P1", P(1), "P2", P(2), "P12", P(3), "P1*P2", P(1) * P(2))
