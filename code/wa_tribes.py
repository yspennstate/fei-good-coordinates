"""W_a test: sum_i Inf_i^a E_i <= C sum_i Inf_i^(a+1) on monotone f.
f_k(x,y) = MAJ_k(y) OR (x AND y >= y0), |y0|=(k-1)/2; each y_j <- Tribes(w,s) on fresh variables.
(1) direct truth-table check of the composition identity (E_x, Inf_x = mu-biased values of f_k)
(2) large-N lower bound of LHS/RHS from the formula."""
import itertools, math, sys
import numpy as np

def h2(t):
    return 0.0 if t <= 0 or t >= 1 else -(t*math.log2(t) + (1-t)*math.log2(1-t))

def wht(v):
    v = v.astype(float).copy(); n = v.size; hh = 1
    while hh < n:
        v = v.reshape(-1, 2*hh); a = v[:, :hh].copy(); b = v[:, hh:].copy()
        v[:, :hh] = a + b; v[:, hh:] = a - b; v = v.reshape(-1); hh *= 2
    return v / n

def E_inf_all(fhat, n):
    w2 = fhat**2; idx = np.arange(fhat.size); out = []
    for i in range(n):
        b = 1 << i; T = idx[(idx & b) == 0]; a = w2[T]; c = w2[T | b]; s = a + c
        m = s > 1e-300; t = np.zeros_like(s); t[m] = c[m]/s[m]
        H = np.where((t > 0) & (t < 1), -(t*np.log2(np.where(t>0,t,1)) + (1-t)*np.log2(np.where(t<1,1-t,1))), 0.0)
        out.append((float((s*H).sum()), float(c.sum())))
    return out

def f_k(x, y, k):
    y0 = [1]*((k-1)//2) + [0]*((k+1)//2)
    maj = sum(y) > k//2
    dom = all(yi >= zi for yi, zi in zip(y, y0))
    return maj or (x and dom)

def tribes(z, w, s):
    return any(all(z[t*w:(t+1)*w]) for t in range(s))

def biased_E_inf_x(k, p):
    """mu-biased orthonormal Fourier of f_k (x unbiased, y_j P(1)=p); E_x, Inf_x."""
    n = k + 1; N = 1 << n
    tab = np.zeros(N)
    for m in range(N):
        bits = [(m >> i) & 1 for i in range(n)]
        tab[m] = 1.0 if f_k(bits[0], bits[1:], k) else -1.0
    # tensor transform: coordinate 0 unbiased, others p-biased
    v = tab.reshape([2]*n[::-1] if False else [2]*n)  # axis order: last axis = bit 0
    v = tab.copy().reshape([2]*n)
    for i in range(n):
        ax = n - 1 - i
        pp = 0.5 if i == 0 else p; qq = 1 - pp; mu = pp - qq; sd = math.sqrt(1 - mu*mu)
        a0 = np.take(v, 0, axis=ax); a1 = np.take(v, 1, axis=ax)
        c0 = qq*a0 + pp*a1; c1 = (qq*a0*(-1-mu) + pp*a1*(1-mu))/sd
        v = np.stack([c0, c1], axis=ax)
    fhat = v.reshape(-1)
    return E_inf_all(fhat, n)

def direct(k, w, s):
    B = w*s; n = 1 + k*B; N = 1 << n
    m = np.arange(N); tab = np.empty(N)
    for mm in range(N):
        bits = [(mm >> i) & 1 for i in range(n)]
        y = [tribes(bits[1+j*B:1+(j+1)*B], w, s) for j in range(k)]
        tab[mm] = 1.0 if f_k(bits[0], y, k) else -1.0
    fhat = wht(tab)
    return E_inf_all(fhat, n)

if __name__ == "__main__":
    print("(1) direct vs formula")
    for (k, w, s) in [(3, 2, 2), (3, 2, 3), (3, 3, 2)]:
        d = direct(k, w, s)
        p = 1 - (1 - 2.0**-w)**s
        b = biased_E_inf_x(k, p)
        print(f"k={k} tribes({w},{s}) p={p:.5f}: direct E_x={d[0][0]:.12f} Inf_x={d[0][1]:.12f}  formula E_x={b[0][0]:.12f} Inf_x={b[0][1]:.12f}")
        # base influence formula check on first base var
        infv_phi = 2.0**-(w-1) * (1 - 2.0**-w)**(s-1)
        print(f"   base var Inf direct={d[1][1]:.12f} formula={b[1][1]*infv_phi:.12f}")
    print("(2) large-N lower bound of LHS/RHS, Ben-Or-Linial tribes width w, s=round(2^w ln2)")
    for k in (3, 5, 7, 9, 11):
        for w in (8, 16, 24, 32, 48):
            s = round(2**w * math.log(2)); p = 1 - (1 - 2.0**-w)**s
            b = biased_E_inf_x(k, p); Ex, Ix = b[0]; rx = Ex/Ix
            infv = 2.0**-(w-1) * (1 - 2.0**-w)**(s-1); Nb = w*s
            line = [f"k={k} w={w} p={p:.6f} r_x={rx:.5f}"]
            for a in (0.0, 0.1, 0.25, 0.5, 1.0):
                base = sum(b[1+j][1]**(1+a) for j in range(k)) * Nb * infv**(1+a)
                lb = Ix**(1+a)*rx / (Ix**(1+a) + base)
                line.append(f"a={a}:{lb:.4f}")
            print("  ".join(line)); sys.stdout.flush()
