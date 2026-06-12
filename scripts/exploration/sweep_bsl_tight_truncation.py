"""D0: bsl + tight truncation thresholds at cv=0.7.

Goal: test whether tighter truncation can dissolve the star foreclosure at bsl (ω=0.50),
where the existing cap=2.5/floor=0.5 (Approach B) failed (max_ten spikes 593/267/192).

Spec:
  - cand: bsl (μ=0.7, ω=0.50, γ=0.10)
  - cv: 0.7 (the cv where star bimodality is clearest at bsl)
  - 3 threshold pairs: (cap=1.3, floor=0.7), (cap=1.5, floor=0.65), (cap=1.75, floor=0.6)
  - 12 etas: full grid {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
  - 5 seeds: {26462-26466}
  - Total: 3 × 12 × 5 = 180 sims, ~15 min wall
  - Regime: socialized_tax (canonical for claim 3)
  - Output: sweep_bsl_tight_truncation_raw.csv

Hypothesis: tight cap creates a "tier of equals" at the boundary, distributing dominance
across multiple banks rather than allowing single-champion lock-in. If so, max_ten
should be bounded and avg_ten should improve toward >3.0.

Schema matches sweep_approach_b_raw.csv for direct comparison.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]
ETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CV = 0.7
WORKERS = 6

# Tight threshold pairs: tighter than cap=2.5/floor=0.5
THRESHOLDS = [
    (1.30, 0.70),  # tight: ~25% of bank draws clamped at cap, ~25% at floor
    (1.50, 0.65),  # moderate: ~12% at cap, ~18% at floor
    (1.75, 0.60),  # loose-ish: ~8% at cap, ~13% at floor
]

CAND = ('bsl', 0.70, 0.50, 0.10)  # (label, mu, omega, gamma)

CHECKPOINT_EVERY = 50
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'sweep_bsl_tight_truncation_raw.csv')

KEYS = ['cand', 'mu', 'omega', 'gamma', 'cv', 'max_factor', 'min_factor',
        'eta', 'seed', 'fiscal_regime',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def run_one(args):
    cand, mu, omega, gamma, cv, max_f, min_f, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = cv
    cfg['equity_max_factor'] = max_f
    cfg['equity_min_factor'] = min_f
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T])
    raw_ids = [b for b in bl if b >= 0]
    runs = []
    if raw_ids:
        prev = raw_ids[0]; rl = 1
        for k in raw_ids[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    avg_ten = sum(runs)/len(runs) if runs else 0
    blc = [s.best_lender_clients[t] for t in range(T) if s.best_lender_clients[t] >= 0]
    avg_cli = sum(blc)/len(blc) if blc else 0
    blf = [s.best_lender_fitness[t] for t in range(T)
           if s.best_lender_fitness[t] is not None and s.best_lender_fitness[t] >= 0]
    avg_fitness = sum(blf)/len(blf) if blf else 0
    bill = float(np.nansum(s.bailout_bill[:T])) if hasattr(s, 'bailout_bill') else 0.0
    def _sum(name):
        arr = getattr(s, name, None)
        if arr is None: return 0
        return int(np.nansum(arr[:T]))
    return {
        'cand': cand, 'mu': mu, 'omega': omega, 'gamma': gamma, 'cv': cv,
        'max_factor': max_f, 'min_factor': min_f,
        'eta': eta, 'seed': seed, 'fiscal_regime': 'socialized_tax',
        'total_bk': _sum('bankruptcy'),
        'shock': _sum('bankruptcies_shock'),
        'rationing': _sum('bankruptcies_rationing'),
        'repay': _sum('bankruptcies_repay'),
        'contagion': _sum('bankruptcies_contagion'),
        'fiscal_deaths': _sum('bankruptcies_fiscal'),
        'zombies': _sum('fire_sale_survivors'),
        'bailout_bill': round(bill, 2),
        'max_ten': max_ten,
        'avg_ten': round(avg_ten, 2),
        'turnovers': max(0, len(runs) - 1),
        'avg_cli': round(avg_cli, 2),
        'avg_fitness': round(avg_fitness, 4),
    }


def write_checkpoint(rows, n_done, n_total):
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in KEYS})
    print(f'  [checkpoint @ {n_done}/{n_total}] wrote {len(rows)} rows', flush=True)


def ms(vals):
    n = len(vals)
    if n == 0: return 0, 0
    m = sum(vals)/n
    if n < 2: return m, 0
    v = sum((x-m)**2 for x in vals)/(n-1)
    return m, math.sqrt(v)


def main():
    cand, mu, omega, gamma = CAND
    jobs = [(cand, mu, omega, gamma, CV, max_f, min_f, e, s)
            for (max_f, min_f) in THRESHOLDS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'D0: bsl tight-truncation sweep (cv={CV})', flush=True)
    print(f'  cand: {cand} (mu={mu}, omega={omega}, gamma={gamma})', flush=True)
    print(f'  thresholds: {THRESHOLDS}', flush=True)
    print(f'  etas: {ETAS}, seeds: {SEEDS}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers', flush=True)
    print('', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % CHECKPOINT_EVERY == 0 or i == n:
                write_checkpoint(rows, i, n)

    print('', flush=True)
    print('=== Per-threshold full stats at eta* (claim 3 optimum) ===', flush=True)
    print(f'  {"cap":>4} {"floor":>5} | {"eta*":>5} | {"bk@0":>6} {"bk*":>6} {"d":>+5} | '
          f'{"max_ten m+/-s":>13} {"s/m":>5} | {"avg_ten":>7} | {"turn":>4} | '
          f'{"shock":>5} {"contag":>6} {"fiscal":>6} {"zom":>5} {"bill":>5}', flush=True)
    print('  ' + '-' * 130, flush=True)
    for max_f, min_f in THRESHOLDS:
        eta_means = {}
        for eta in ETAS:
            cells = [r for r in rows if abs(r['max_factor']-max_f)<1e-6
                     and abs(r['min_factor']-min_f)<1e-6 and abs(r['eta']-eta)<1e-6]
            if cells:
                eta_means[eta] = ms([r['total_bk'] for r in cells])[0]
        if not eta_means: continue
        bk0 = eta_means.get(0.0, 0)
        eta_star = min(eta_means, key=eta_means.get)
        bk_star = eta_means[eta_star]
        delta = bk_star - bk0
        recs_star = [r for r in rows if abs(r['max_factor']-max_f)<1e-6
                     and abs(r['min_factor']-min_f)<1e-6 and abs(r['eta']-eta_star)<1e-6]
        mt_m, mt_s = ms([r['max_ten'] for r in recs_star])
        av_m, _ = ms([r['avg_ten'] for r in recs_star])
        tn_m, _ = ms([r['turnovers'] for r in recs_star])
        sh_m, _ = ms([r['shock'] for r in recs_star])
        ct_m, _ = ms([r['contagion'] for r in recs_star])
        fi_m, _ = ms([r['fiscal_deaths'] for r in recs_star])
        zm_m, _ = ms([r['zombies'] for r in recs_star])
        bl_m, _ = ms([r['bailout_bill'] for r in recs_star])
        sm = mt_s/mt_m if mt_m else 0
        print(f'  {max_f:>4.2f} {min_f:>5.2f} | {eta_star:>5.2f} | {bk0:>6.0f} {bk_star:>6.0f} {delta:>+5.0f} | '
              f'{mt_m:>4.0f}+/-{mt_s:<4.0f}    {sm:>5.2f} | {av_m:>5.1f}   | {tn_m:>4.0f} | '
              f'{sh_m:>5.0f} {ct_m:>6.0f} {fi_m:>6.0f} {zm_m:>5.0f} {bl_m:>5.0f}', flush=True)

    print('', flush=True)
    print('=== Per-threshold full eta-profile (max_ten + avg_ten) ===', flush=True)
    for max_f, min_f in THRESHOLDS:
        print(f'  cap={max_f}, floor={min_f}:', flush=True)
        print(f'    {"eta":>5} | {"bk m+/-s":>13} | {"max_ten m+/-s":>13} {"s/m":>5} | {"avg_ten":>7} | {"min-max/seed":>13}', flush=True)
        for eta in ETAS:
            recs = [r for r in rows if abs(r['max_factor']-max_f)<1e-6
                    and abs(r['min_factor']-min_f)<1e-6 and abs(r['eta']-eta)<1e-6]
            if not recs: continue
            bk_m,bk_s = ms([r['total_bk'] for r in recs])
            mt_m,mt_s = ms([r['max_ten'] for r in recs])
            av_m,_ = ms([r['avg_ten'] for r in recs])
            mts = sorted([int(r['max_ten']) for r in recs])
            sm = mt_s/mt_m if mt_m else 0
            print(f'    {eta:>5.2f} | {bk_m:>5.0f}+/-{bk_s:<5.0f} | {mt_m:>4.0f}+/-{mt_s:<4.0f}    {sm:>5.2f} | {av_m:>5.1f}   | {min(mts):>4}-{max(mts):<5}', flush=True)


if __name__ == '__main__':
    main()
