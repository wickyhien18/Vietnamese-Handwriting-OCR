"""Kiem chung so hoc cho Phan B/C (chi dung numpy + thu vien chuan). Chay: python3 verify_partBC.py"""
import itertools, math, random
import numpy as np

# ---- C1: Levenshtein ban/bạn voi truy vet, quy tac uu tien: cheo, thua, thieu
def lev_table(a,b):
    n,m=len(a),len(b); d=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0]=i
    for j in range(m+1): d[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i][j]=min(d[i-1][j-1]+(a[i-1]!=b[j-1]), d[i-1][j]+1, d[i][j-1]+1)
    return d
print("C1 bang 'ban' (hang) x 'bạn' (cot):")
for r in lev_table("ban","bạn"): print(r)

def all_alignments(a,b):
    d=lev_table(a,b); out=[]
    def rec(i,j,ops):
        if i==0 and j==0: out.append(tuple(reversed(ops))); return
        if i>0 and j>0 and d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
            rec(i-1,j-1,ops+['M' if a[i-1]==b[j-1] else 'S'])
        if i>0 and d[i][j]==d[i-1][j]+1: rec(i-1,j,ops+['D'])
        if j>0 and d[i][j]==d[i][j-1]+1: rec(i,j-1,ops+['I'])
    rec(len(a),len(b),[]); return d[len(a)][len(b)],out
print("C2 'ab'->'ba':", all_alignments("ab","ba"))

# ---- C3: tau
for r in (1,2,5): print("C3 ch/cb=%d tau=%.4f"%(r, r/(r+1)))

# ---- C4/B3: CTC brute force
def collapse(p):
    out=[];prev=None
    for x in p:
        if x!=prev and x!=0: out.append(x)
        prev=x
    return tuple(out)
y=np.array([[0.6,0.4],[0.6,0.4]])  # cot0=blank, cot1='a'
tot={}
for pi in itertools.product([0,1],repeat=2):
    pr=np.prod([y[t,pi[t]] for t in range(2)]); tot[collapse(pi)]=tot.get(collapse(pi),0)+pr
print("C4 P(rong)=%.2f P('a')=%.2f"%(tot[()],tot[(1,)]))

# ---- B3: forward recursion == brute force (random)
def ctc_forward(y,l):
    T=y.shape[0]; lp=[0]+[x for c in l for x in (c,0)]; S=len(lp)
    a=np.zeros((T,S)); a[0,0]=y[0,0]
    if S>1: a[0,1]=y[0,lp[1]]
    for t in range(1,T):
        for s in range(S):
            v=a[t-1,s]+(a[t-1,s-1] if s>=1 else 0)
            if s>=2 and lp[s]!=0 and lp[s]!=lp[s-2]: v+=a[t-1,s-2]
            a[t,s]=v*y[t,lp[s]]
    return a[-1,-1]+(a[-1,-2] if S>1 else 0)
rng=np.random.default_rng(0); worst=0
for _ in range(200):
    T=int(rng.integers(3,8)); K=3
    yy=rng.dirichlet(np.ones(K),size=T); l=tuple(int(x) for x in rng.integers(1,K,size=int(rng.integers(1,4))))
    bf=sum(np.prod([yy[t,pi[t]] for t in range(T)]) for pi in itertools.product(range(K),repeat=T) if collapse(pi)==l)
    worst=max(worst,abs(bf-ctc_forward(yy,l)))
print("B3 forward vs brute force, max |diff| over 200 ca:",worst)

# ---- B4: Viterbi == brute force
rng=np.random.default_rng(1); worst=0
for _ in range(200):
    n=int(rng.integers(2,6)); K=int(rng.integers(2,4))
    em=rng.random((n,K)); tr=rng.random((n-1,K,K))
    sc=lambda p: np.log(em[0,p[0]])+sum(np.log(tr[i-1,p[i-1],p[i]])+np.log(em[i,p[i]]) for i in range(1,n))
    best=max(sc(p) for p in itertools.product(range(K),repeat=n))
    v=np.log(em[0]).copy()
    for i in range(1,n): v=np.max(v[:,None]+np.log(tr[i-1]),axis=0)+np.log(em[i])
    worst=max(worst,abs(best-v.max()))
print("B4 viterbi vs brute force, max |diff|:",worst)

# ---- Hoeffding
for m in (5,9): print("Hoeffding m=%d delta=.05 width=%.3f"%(m,math.sqrt(math.log(2/0.05)/(2*m))))

# ---- C5: coverage mo phong
def sim(m=9,n=70,rho=0.3,mu=0.12,reps=1500,B=400,seed=7):
    rng=np.random.default_rng(seed); cov={'hoeffding':0,'boot_dong':0,'boot_doan':0}; w={'hoeffding':[],'boot_dong':[],'boot_doan':[]}
    sd=math.sqrt(rho)*0.06; sdn=math.sqrt(1-rho)*0.06
    for _ in range(reps):
        pe=mu+rng.normal(0,sd,m); x=np.clip(pe[:,None]+rng.normal(0,sdn,(m,n)),0,1)
        pm=x.mean(1); gm=x.mean()
        h=math.sqrt(math.log(40)/(2*m)); lo,hi=gm-h,gm+h
        cov['hoeffding']+=lo<=mu<=hi; w['hoeffding'].append(hi-lo)
        flat=x.ravel(); bl=np.array([rng.choice(flat,flat.size).mean() for _ in range(B)])
        lo,hi=np.percentile(bl,[2.5,97.5]); cov['boot_dong']+=lo<=mu<=hi; w['boot_dong'].append(hi-lo)
        idx=rng.integers(0,m,(B,m)); bp=pm[idx].mean(1)   # cac doan cung do dai
        lo,hi=np.percentile(bp,[2.5,97.5]); cov['boot_doan']+=lo<=mu<=hi; w['boot_doan'].append(hi-lo)
    return {k:(cov[k]/reps,float(np.mean(w[k]))) for k in cov}
print("C5 (m=9 doan, n=70 dong/doan, rho=0.3):",sim())
print("C5 (rho=0.0):",sim(rho=0.0,reps=800))
# design effect
print("Design effect n=70 rho=0.3:",1+(70-1)*0.3)

# ---- C7: ECE minh hoa
def ece(conf,correct,bins=10):
    e=0;N=len(conf)
    for b in range(bins):
        mk=(conf>b/bins)&(conf<=(b+1)/bins)
        if mk.any(): e+=mk.sum()/N*abs(conf[mk].mean()-correct[mk].mean())
    return e
rng=np.random.default_rng(3); z=rng.normal(0,1,20000)*2.0+1.5
true_p=1/(1+np.exp(-z)); corr=(rng.random(20000)<true_p).astype(float)
over=1/(1+np.exp(-z*2.0))   # mo hinh qua tu tin
print("C7 ECE truoc (T=1): %.4f"%ece(over,corr))
best=min(((ece(1/(1+np.exp(-z*2.0/T)),corr),T) for T in np.arange(0.5,4.01,0.05)))
print("C7 ECE sau (T=%.2f): %.4f"%(best[1],best[0]))

# ---- C5 nhieu hat giong (bao cao: trung binh +- do lech chuan giua 5 hat giong; rieng bootstrap 300 mau, 600 lan lap/hat giong)
if __name__ == "__main__":
    for rho in (0.3, 0.0):
        rs = [sim(rho=rho, reps=600, B=300, seed=s) for s in (7, 11, 23, 42, 101)]
        for m in ("hoeffding", "boot_dong", "boot_doan"):
            c = np.array([float(r[m][0]) for r in rs]); w = np.array([float(r[m][1]) for r in rs])
            print("C5 5 seeds rho=%.1f %-10s cover %.1f%% +- %.1f  width %.4f" % (rho, m, 100*c.mean(), 100*c.std(ddof=1), w.mean()))
