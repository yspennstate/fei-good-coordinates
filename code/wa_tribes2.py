import math, sys
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("w", "wa_tribes.py"); w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
# (1') base-variable check with the 4pq normalisation
for (k, ww, s) in [(3, 2, 2), (3, 2, 3), (3, 3, 2)]:
    d = w.direct(k, ww, s); p = 1 - (1 - 2.0**-ww)**s; b = w.biased_E_inf_x(k, p)
    infv = 2.0**-(ww-1) * (1 - 2.0**-ww)**(s-1)
    print(f"k={k} tribes({ww},{s}) base Inf direct={d[1][1]:.12f} formula={b[1][1]/(4*p*(1-p))*infv:.12f}")
# (2') large widths in logs: p = 1-exp(s*log1p(-2^-w)); Inf_v(phi) = 2^-(w-1) exp((s-1) log1p(-2^-w))
for k in (3, 7, 11, 15):
    for ww in (64, 128, 256, 512):
        s = round(2.0**ww * math.log(2)); L = math.log1p(-2.0**-ww)
        p = 1 - math.exp(s*L); b = w.biased_E_inf_x(k, p); Ex, Ix = b[0]; rx = Ex/Ix
        log_infv = -(ww-1)*math.log(2) + (s-1)*L; log_Nb = math.log(ww) + math.log(s)
        piv = [b[1+j][1]/(4*p*(1-p)) for j in range(k)]
        out = [f"k={k} w={ww} r_x={rx:.5f}"]
        for a in (0.05, 0.1, 0.25):
            base = sum(q**(1+a) for q in piv) * math.exp(log_Nb + (1+a)*log_infv)
            out.append(f"a={a}:{Ix**(1+a)*rx/(Ix**(1+a)+base):.4f}")
        print("  ".join(out)); sys.stdout.flush()
