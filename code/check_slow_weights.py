"""Check: g slowly varying, balanced monotone psi, Q_g(Maj_m o psi) -> G(psi); and E,Inf scale by tau_m exactly."""
from math import log, log2
import sys
def h2(q): return 0.0 if q<=0 or q>=1 else -q*log2(q)-(1-q)*log2(1-q)
def spec(T,n):
    N=1<<n; F=[1-2*((T>>p)&1) for p in range(N)]
    # fast WHT
    a=F[:]; h=1
    while h<N:
        for i in range(0,N,2*h):
            for j in range(i,i+h):
                x,y=a[j],a[j+h]; a[j],a[j+h]=x+y,x-y
        h*=2
    return [(v/N)**2 for v in a]
def EI(mu,n):
    N=1<<n; out=[]
    for i in range(n):
        b=1<<i; inf=sum(mu[S] for S in range(N) if S&b); E=0.0
        for S in range(N):
            if S&b: continue
            s=mu[S]+mu[S|b]
            if s>0: E+=s*h2(mu[S|b]/s)
        out.append((E,inf))
    return out
def Q(ei,g):
    num=sum(g(I)*E for E,I in ei if I>1e-15); den=sum(g(I)*I for E,I in ei if I>1e-15); return num/den
n=4; T=0xeac8
f=[(T>>p)&1 for p in range(16)]
print("balanced",sum(f)==8,"monotone",all(f[p]<=f[p|(1<<b)] for p in range(16) for b in range(4)))
mu=spec(T,n); ei=EI(mu,n)
for name,g in [("Q0",lambda t:1.0),("Q1 ln",lambda t:1/log(1/t)),("Q1 log2",lambda t:1/log2(1/t))]:
    print(name,round(Q(ei,g),6))
# Maj3 o psi on 12 variables, brute force
n2=12; N2=1<<n2; tt=0
for p in range(N2):
    v=[f[(p>>(4*j))&15] for j in range(3)]
    if sum(v)>=2: tt|=1<<p
mu2=spec(tt,n2); ei2=EI(mu2,n2)
tau=0.5
print("scale check max err",max(abs(ei2[4*j+i][0]-tau*ei[i][0])+abs(ei2[4*j+i][1]-tau*ei[i][1]) for j in range(3) for i in range(4)))
for name,g in [("Q0",lambda t:1.0),("Q1 ln",lambda t:1/log(1/t))]:
    print("Maj3o",name,round(Q(ei2,g),6))
# limit via scaling with tau_m ~ sqrt(2/(pi m))
from math import comb
for m in [3,11,101,10001,10**6+1]:
    tm=comb(m-1,(m-1)//2)/2**(m-1)
    print(m,round(Q([(tm*E,tm*I) for E,I in ei],lambda t:1/log(1/t)),6))
