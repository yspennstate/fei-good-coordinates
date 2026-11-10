"""(c) pointwise: analytic -V_alpha = f(a)/s^2 - f'(a)[1/s - 1/(1+2a)] vs finite difference, and the claimed bound >= f(b)/s^2."""
import mpmath as mp
mp.mp.dps = 30
Phi = lambda a, b: a * mp.log((a + b) / a, 2) + b * mp.log((a + b) / b, 2) if a > 0 else b * 0
F = lambda u: Phi(u * u, 1)
f = lambda t: F(2 * t + 1)
fp = lambda t: mp.diff(f, t)
W = lambda a: mp.quad(lambda t: 2 * f(t) / (1 + 2 * t) ** 2, [a, a + 10, mp.inf])
V = lambda a, b: f(a) * (1 / (1 + a + b) - 1 / (1 + 2 * a)) + W(a)
worst = None
for a in [0, 0.01, 0.1, 0.3, 1, 2, 5, 20, 100]:
    for frac in [0, 0.1, 0.5, 0.9, 1.0]:
        a, b = mp.mpf(a), mp.mpf(a) * frac
        s = 1 + a + b
        an = f(a) / s ** 2 - fp(a) * (1 / s - 1 / (1 + 2 * a))
        fd = -mp.diff(lambda x: V(x, b), a) if a > 0 else None
        gap = an - f(b) / s ** 2
        if worst is None or gap < worst[0]:
            worst = (gap, a, b)
        if fd is not None and abs(fd - an) > 1e-12:
            print("MISMATCH", a, b, fd, an)
print("min gap -V_alpha - f(beta)/s^2 =", mp.nstr(worst[0], 6), "at alpha,beta =", worst[1], worst[2])
