#!/usr/bin/env python3
"""Final three-axis combined recomputation (the single set of numbers reported in the paper).

Inputs are the per_seed CSVs for the three axes (column-name differences are handled):
  - keep: paper/data/keep_per_seed_v2_final.csv   (level column = keep, values 1.0/0.5/0.3/0.1)
  - time: paper/data/time_per_seed_v2_final.csv   (level column = tq_us, values 0/1000/5000/10000)
  - patch: paper/data/patch_per_seed.csv          (level column = patch, patch counts 96/64/32/16)
Shared core columns: dataset, scene, level, seed, status, mpe (present in all three axes).

Produces the final paper numbers:
  - per-cell three classes (valid / numerical_crash / catastrophic MPE>100%) + failure rate + Wilson 95% CI
  - fraction of cells whose failures are hidden by a naive valid-only mean +/- std (three axes combined)
  - bootstrap 95% CI for the single-run misleading metric Q1 (under-reported failures)
Usage: python scripts/final_recompute.py  (run directly once the CSVs are in place)
"""
import csv, math, statistics as st, os, random
from collections import defaultdict

DATA = "paper/data"
THETA = 100.0
AXES = [
    ("event_density(keep)", f"{DATA}/keep_per_seed_v2_final.csv",  "keep",  ["1.0","0.5","0.3","0.1"]),
    ("timestamp(tq_us)",    f"{DATA}/time_per_seed_v2_final.csv",  "tq_us", ["0","1000","5000","10000"]),
    ("visual_budget(patch)",f"{DATA}/patch_per_seed_v2_final.csv", "patch", ["96","64","32","16"]),  # patch-v2 third column = patch
]

def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan")
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=(z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
    return max(0,(c-h))*100, min(1,(c+h))*100

def classify(status, mpe):
    if status=="crash" or mpe is None: return "crash"
    if mpe>THETA: return "catastrophic"
    return "valid"

def load(path, levelcol):
    """Read a CSV, handling column-name differences. Returns cells[(ds,scene,level)] = [(class,mpe),...]."""
    if not os.path.isfile(path):
        return None
    cells=defaultdict(list)
    for r in csv.DictReader(open(path)):
        mpe=float(r["mpe"]) if r["mpe"] else None
        cells[(r["dataset"],r["scene"],r[levelcol])].append((classify(r["status"],mpe), mpe))
    return cells

def main():
    grand={"cells":0,"valid":0,"crash":0,"cat":0,"naive_hidden":0,"partial":0}
    missing=[]
    all_cells={}  # used for naive / bootstrap
    for name,path,lc,lvls in AXES:
        cells=load(path,lc)
        if cells is None:
            missing.append(path); print(f"[WARN] missing: {path} (skipping this axis)"); continue
        print(f"\n{'='*90}\n{name} ({len([k for k in cells])}cell)\n{'='*90}")
        print(f"{'seq':28} {'lvl':>6} {'V/C/X':>8} {'fail%':>6} {'Wilson95%':>13} {'med':>7} {'worst':>8}")
        for (ds,sc) in sorted(set((k[0],k[1]) for k in cells)):
            for lv in lvls:
                runs=cells.get((ds,sc,lv))
                if not runs: continue
                n=len(runs); nc=sum(c=="crash" for c,_ in runs); nx=sum(c=="catastrophic" for c,_ in runs)
                nv=n-nc-nx; nf=nc+nx
                vmpes=[m for c,m in runs if c=="valid"]
                med=st.median(vmpes) if vmpes else float('nan')
                wst=max((m for _,m in runs if m is not None), default=float('nan'))
                lo,hi=wilson(nf,n)
                grand["cells"]+=1; grand["valid"]+=nv; grand["crash"]+=nc; grand["cat"]+=nx
                if nf>0: grand["naive_hidden"]+=1
                if 0<nf<n: grand["partial"]+=1
                all_cells[(name,ds,sc,lv)]=runs
                fl=" †" if nv<3 else ""
                print(f"{ds+'/'+sc:28} {lv:>6} {nv}/{nc}/{nx:>2} {100*nf/n:>5.0f}% [{lo:>4.0f},{hi:>4.0f}]% {med:>7.2f} {wst:>8.2f}{fl}")

    print(f"\n{'='*90}\nFinal paper numbers (three axes combined, the only reported version)\n{'='*90}")
    if missing:
        print(f"[WARN] note: the following CSVs are missing, so the numbers below are incomplete: {missing}")
    nc=grand["cells"]
    print(f"total cells: {nc}")
    print(f"valid={grand['valid']} numerical_crash={grand['crash']} catastrophic_completion={grand['cat']}")
    print(f"cells with >=1 failure hidden by a naive valid-only mean+/-std: {grand['naive_hidden']}/{nc} "
          f"({100*grand['naive_hidden']/nc:.0f}%)")
    print(f"  of which partial-failure cells: {grand['partial']}")

    # Q1 bootstrap: within partial-failure cells, probability that a single draw
    # picks a non-crashing seed (i.e. under-reports the failure).
    partial=[runs for runs in all_cells.values() if 0<sum(c!='valid' for c,_ in runs)<len(runs)]
    if partial:
        miss_probs=[sum(c=='valid' for c,_ in runs)/len(runs) for runs in partial]
        mean_miss=100*sum(miss_probs)/len(miss_probs)
        rng=random.Random(42); boots=[]
        for _ in range(2000):
            samp=[miss_probs[rng.randrange(len(miss_probs))] for _ in miss_probs]
            boots.append(100*sum(samp)/len(samp))
        boots.sort()
        print(f"\nQ1 single-run under-reported failures (mean over partial cells): {mean_miss:.0f}% "
              f"(bootstrap 95% [{boots[50]:.0f},{boots[1949]:.0f}]%, {len(partial)} partial cells)")

    # ===== Q2/Q3: from the same final three-axis data, reload per-run MPE with seeds =====
    # all_cells only stored (class, mpe); Q2/Q3 need per-seed alignment, so reload the CSV with seeds.
    def load_seed(path, lc):
        if not os.path.isfile(path): return None
        d=defaultdict(dict)  # (ds,sc,lv) -> {seed: mpe or None}
        for r in csv.DictReader(open(path)):
            d[(r["dataset"],r["scene"],r[lc])][int(r["seed"])]=float(r["mpe"]) if r["mpe"] else None
        return d

    # Q2 ranking flips: per axis, rank sequences by a single seed at the clean level,
    # then measure the fraction of sequence pairs whose order flips between seed pairs.
    # Q3 direction errors: per axis, fraction of single (clean_seed, degraded_seed)
    # comparisons whose direction is opposite to the repeated-run median reference direction.
    print(f"\n{'='*90}\nQ2/Q3 (same final three-axis data source)\n{'='*90}")
    q2_meds=[]; q3_all=0; q3_opp=0
    for name,path,lc,lvls in AXES:
        d=load_seed(path,lc)
        if d is None: continue
        clean=lvls[0]; deg=lvls[1]  # clean level + first degraded level
        seqs=sorted(set((ds,sc) for (ds,sc,lv) in d if lv==clean))
        seeds=sorted({s for k in d for s in d[k]})
        # Q2: over all seed pairs, fraction of sequence pairs whose clean ranking flips
        flips=[]
        for i in range(len(seeds)):
            for j in range(i+1,len(seeds)):
                sa,sb=seeds[i],seeds[j]
                pairs=disagree=0
                for a in range(len(seqs)):
                    for b in range(a+1,len(seqs)):
                        ma_a=d.get((*seqs[a],clean),{}).get(sa); ma_b=d.get((*seqs[b],clean),{}).get(sa)
                        mb_a=d.get((*seqs[a],clean),{}).get(sb); mb_b=d.get((*seqs[b],clean),{}).get(sb)
                        if None in (ma_a,ma_b,mb_a,mb_b): continue
                        pairs+=1
                        if (ma_a<ma_b)!=(mb_a<mb_b): disagree+=1
                if pairs: flips.append(100*disagree/pairs)
        if flips:
            flips.sort(); med=flips[len(flips)//2]
            q2_meds.append((name,med,flips[0],flips[-1]))
        # Q3: clean vs deg, single-run direction vs multi-run median direction
        for (ds,sc) in seqs:
            cm=[d[(ds,sc,clean)][s] for s in seeds if d.get((ds,sc,clean),{}).get(s) is not None]
            dm=[d[(ds,sc,deg)][s]   for s in seeds if d.get((ds,sc,deg),{}).get(s) is not None]
            if len(cm)<1 or len(dm)<1: continue
            ref_worse = st.median(dm) > st.median(cm)  # reference: is the degraded level worse?
            for ci in cm:
                for di in dm:
                    q3_all+=1
                    if (di>ci)!=ref_worse: q3_opp+=1
    for name,med,lo,hi in q2_meds:
        print(f"Q2 {name}: ranking flips median {med:.0f}% (range {lo:.0f}-{hi:.0f}%)")
    if q3_all:
        rng=random.Random(7); boots=[]
        # bootstrap resampling over all comparisons
        comps=q3_opp; tot=q3_all
        print(f"Q3 single-run direction error (three axes combined): {100*comps/tot:.0f}% ({comps}/{tot} comparisons)")
    print(f"\n-> Q1/Q2/Q3 all come from the same final three-axis CSVs. This is the only version used in the paper text/abstract/tables.")

if __name__=="__main__":
    main()
