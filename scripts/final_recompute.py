#!/usr/bin/env python3
"""三轴合并最终重算 (论文唯一认这一版数字, 2026-06-23).

输入三个轴的 per_seed CSV (列名差异已兼容):
  - keep: paper/data/keep_per_seed_v2_final.csv   (stress-v2, level列名=keep, 值1.0/0.5/0.3/0.1)
  - time: paper/data/time_per_seed_v2_final.csv   (stress-v2, level列名=tq_us, 值0/1000/5000/10000)
  - patch: paper/data/patch_per_seed.csv    (旧但有效, level列名=keep实为patch数96/64/32/16, 有reason列)
共同核心列: dataset,scene,level,seed,status,mpe (三轴都有)。

产出论文最终数字:
  - 每cell三类(valid/numerical_crash/catastrophic MPE>100%) + 失败率 + Wilson 95% CI
  - naive valid-only mean±std 抹掉失败的cell占比 (三轴合并, 替换旧37/96)
  - 单次评测误导Q1(漏报失败) bootstrap 95% CI
口径完全沿用 recompute_failure.py / recompute_v2.py。
用法: python script/final_recompute.py  (CSV齐了直接跑)
"""
import csv, math, statistics as st, os, random
from collections import defaultdict

DATA = "paper/data"
THETA = 100.0
AXES = [
    ("event_density(keep)", f"{DATA}/keep_per_seed_v2_final.csv",  "keep",  ["1.0","0.5","0.3","0.1"]),
    ("timestamp(tq_us)",    f"{DATA}/time_per_seed_v2_final.csv",  "tq_us", ["0","1000","5000","10000"]),
    ("visual_budget(patch)",f"{DATA}/patch_per_seed_v2_final.csv", "patch", ["96","64","32","16"]),  # patch-v2 第3列名=patch
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
    """读CSV, 兼容列名差异. 返回 cells[(ds,scene,level)] = [(class,mpe),...]."""
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
    all_cells={}  # 用于naive/bootstrap
    for name,path,lc,lvls in AXES:
        cells=load(path,lc)
        if cells is None:
            missing.append(path); print(f"⚠️ 缺失: {path} (跳过该轴)"); continue
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

    print(f"\n{'='*90}\n论文最终数字 (三轴合并, 唯一认这版)\n{'='*90}")
    if missing:
        print(f"⚠️ 注意: 以下CSV缺失, 下列数字不完整(待补齐): {missing}")
    nc=grand["cells"]
    print(f"总cell数: {nc}")
    print(f"valid={grand['valid']} numerical_crash={grand['crash']} catastrophic_completion={grand['cat']}")
    print(f"naive valid-only mean±std 抹掉≥1失败的cell: {grand['naive_hidden']}/{nc} "
          f"({100*grand['naive_hidden']/nc:.0f}%)  [替换旧37/96(39%)]")
    print(f"  其中 partial-failure cell: {grand['partial']}")

    # Q1 bootstrap: partial-failure cell里, 单次抽中不崩seed(漏报失败)的概率
    partial=[runs for runs in all_cells.values() if 0<sum(c!='valid' for c,_ in runs)<len(runs)]
    if partial:
        miss_probs=[sum(c=='valid' for c,_ in runs)/len(runs) for runs in partial]
        mean_miss=100*sum(miss_probs)/len(miss_probs)
        rng=random.Random(42); boots=[]
        for _ in range(2000):
            samp=[miss_probs[rng.randrange(len(miss_probs))] for _ in miss_probs]
            boots.append(100*sum(samp)/len(samp))
        boots.sort()
        print(f"\nQ1 单次评测漏报失败(partial cell平均): {mean_miss:.0f}% "
              f"(bootstrap95% [{boots[50]:.0f},{boots[1949]:.0f}]%, {len(partial)}个partial cell)")

    # ===== Q2/Q3: 用同一final三轴数据, 重新加载带seed的per-run MPE =====
    # all_cells只存了(class,mpe), Q2/Q3需要per-seed对齐, 重新读CSV带seed.
    def load_seed(path, lc):
        if not os.path.isfile(path): return None
        d=defaultdict(dict)  # (ds,sc,lv) -> {seed: mpe或None}
        for r in csv.DictReader(open(path)):
            d[(r["dataset"],r["scene"],r[lc])][int(r["seed"])]=float(r["mpe"]) if r["mpe"] else None
        return d

    # Q2 排名翻转: 每轴, clean档下用单seed给序列排名, 所有seed对之间序列对顺序翻转的比例.
    # Q3 方向误判: 每轴, 单次(clean_seed, degraded_seed)对比方向 vs repeated-run中位参考方向, 相反的比例.
    print(f"\n{'='*90}\nQ2/Q3 (同final三轴数据源)\n{'='*90}")
    q2_meds=[]; q3_all=0; q3_opp=0
    for name,path,lc,lvls in AXES:
        d=load_seed(path,lc)
        if d is None: continue
        clean=lvls[0]; deg=lvls[1]  # clean档 + 第一退化档
        seqs=sorted(set((ds,sc) for (ds,sc,lv) in d if lv==clean))
        seeds=sorted({s for k in d for s in d[k]})
        # Q2: 所有seed对, clean排名翻转的序列对比例
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
        # Q3: clean vs deg 单次方向 vs 多run中位方向
        for (ds,sc) in seqs:
            cm=[d[(ds,sc,clean)][s] for s in seeds if d.get((ds,sc,clean),{}).get(s) is not None]
            dm=[d[(ds,sc,deg)][s]   for s in seeds if d.get((ds,sc,deg),{}).get(s) is not None]
            if len(cm)<1 or len(dm)<1: continue
            ref_worse = st.median(dm) > st.median(cm)  # 参考: 退化更差?
            for ci in cm:
                for di in dm:
                    q3_all+=1
                    if (di>ci)!=ref_worse: q3_opp+=1
    for name,med,lo,hi in q2_meds:
        print(f"Q2 {name}: 排名翻转 median {med:.0f}% (范围{lo:.0f}-{hi:.0f}%)")
    if q3_all:
        rng=random.Random(7); boots=[]
        # bootstrap对所有比较重采样
        comps=q3_opp; tot=q3_all
        print(f"Q3 单次方向误判(三轴合并): {100*comps/tot:.0f}% ({comps}/{tot}个comparison)")
    print(f"\n→ Q1/Q2/Q3 全部来自同一final三轴CSV。论文正文/摘要/表只认这版。")

if __name__=="__main__":
    main()
