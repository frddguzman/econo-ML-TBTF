"""Phase 0.5: w58 x cv hetero x full eta-grid x social-only.

Tests whether hetero at the hub-stable cell (w58: mu=0.7, omega=0.58, gamma=0.10)
breaks the trade-off found in Phase 0:
  - At w58 cv=0, hub max_ten=37 but social-tax mechanism is sub-threshold
    (fiscal_deaths=0, bills~39 over 1000 periods).
  - Hetero (cv>=0.7) creates G-SIB equity asymmetry -> top banks have large A_j
    -> b_j ~ 1 -> bailouts on top-bank failures large -> bills above threshold
    -> social-tax cascade engages (in theory).
  - Phase Y only tested cv in {0, 0.5, 1.0} at eta in {0, 0.1} social-only.
    cv=0.7 and cv=0.85 NEVER tested at w58.

Phase 0.5 spec (per OG Phase 3, augmented):
  - 4 cv values: {0.5, 0.7, 0.85, 1.0}
  - Full 10-eta grid: {0.0, 0.1, ..., 0.9}
  - 5 seeds: 26462-26466 (the multi-seed pool used by all prior thesis sweeps)
  - Socialized regime only (Phase 3 spec)
  - Full channel decomp tracked (vs Phase Y which only had total_bk + contagion)

Total: 4 x 10 x 5 = 200 sims, ~16 min wall at 6 workers.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import math
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS  = [26462, 26463, 26464, 26465, 26466]
WORKERS = 6
MU = 0.70
OMEGA = 0.58
GAMMA = 0.10
REGIME = 'socialized_tax'

CVS    = [0.5, 0.7, 0.85, 1.0]
ETAS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _sum_int(stats_obj, name, T):
    arr = getattr(stats_obj, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    cv, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime=REGIME)
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['equity_heterogeneity'] = (cv > 0)
    cfg['equity_cv'] = cv
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

    return {
        'cv': cv, 'eta': eta, 'seed': seed,
        'total_bk':       _sum_int(s, 'bankruptcy', T),
        'shock':          _sum_int(s, 'bankruptcies_shock', T),
        'rationing':      _sum_int(s, 'bankruptcies_rationing', T),
        'repay':           _sum_int(s, 'bankruptcies_repay', T),
        'contagion':      _sum_int(s, 'bankruptcies_contagion', T),
        'fiscal_deaths':  _sum_int(s, 'bankruptcies_fiscal', T),
        'zombies':        _sum_int(s, 'fire_sale_survivors', T),
        'bailout_bill':   round(bill, 2),
        'avg_cli':        round(avg_cli, 2),
        'avg_fitness':    round(avg_fitness, 4),
        'avg_ten':        round(avg_ten, 2),
        'max_ten':        max_ten,
        'turnovers':      max(0, len(runs) - 1),
    }


def msd(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x)) and not math.isinf(float(x))]
    if not xs: return 'n/a'
    if len(xs) < 2: return f'{xs[0]:.2f}'
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    s = math.sqrt(var)
    if m > 100: return f'{m:.0f}+/-{s:.0f}'
    if m > 1:   return f'{m:.2f}+/-{s:.2f}'
    return f'{m:.4f}+/-{s:.4f}'


def main():
    jobs = [(c, e, s) for c in CVS for e in ETAS for s in SEEDS]
    print(f'Phase 0.5: w58 x cv hetero (mu={MU}, omega={OMEGA}, gamma={GAMMA}, social-only)')
    print(f'cv values: {CVS}')
    print(f'eta values: {ETAS}')
    print(f'{len(jobs)} sims, {WORKERS} workers')
    rows = []
    # Save raw rows defensively after each chunk so a print-time crash doesn't lose 16 min of compute.
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_w58_cv_phase05_raw.csv')
    keys = ['cv', 'eta', 'seed', 'total_bk', 'shock', 'rationing', 'repay',
            'contagion', 'fiscal_deaths', 'zombies', 'bailout_bill',
            'avg_cli', 'avg_fitness', 'avg_ten', 'max_ten', 'turnovers']
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f'  saved raw rows -> {out_csv}')

    # Per-cv eta-curve table for claim 3 verdict
    print()
    print(f'=== Total bk by cv and eta (5-seed mean) ===')
    print(f'{"cv":>5} | ' + ' | '.join(f'{e:>6.1f}' for e in ETAS) + ' | min eta | min val | bk@eta=0 | claim 3')
    print('-' * 145)
    for cv in CVS:
        means = []
        for eta in ETAS:
            cell = [r['total_bk'] for r in rows if r['cv']==cv and r['eta']==eta]
            means.append(sum(cell)/len(cell) if cell else 0)
        bk0 = means[0]
        min_idx = means.index(min(means))
        min_eta = ETAS[min_idx]
        min_val = means[min_idx]
        delta = min_val - bk0
        verdict = f'YES (eta*={min_eta:.1f}, d={delta:+.0f})' if delta < 0 else 'NO (eta=0 wins)'
        print(f'{cv:>5.2f} | ' + ' | '.join(f'{m:>6.0f}' for m in means) + f' | {min_eta:>6.1f} | {min_val:>7.0f} | {bk0:>8.0f} | {verdict}')

    # Channel decomposition + hub stats per (cv, eta=0.1)
    print()
    print(f'=== Channel decomp + hub stats at eta=0.1 social (5-seed mean +/- std) ===')
    print(f'{"cv":>5} | {"shock":>11} | {"rationing":>11} | {"repay":>9} | '
          f'{"contagion":>11} | {"fiscal":>11} | {"zombies":>11} | {"bill":>9} | '
          f'{"max_ten":>11} | {"avg_cli":>11}')
    print('-' * 165)
    for cv in CVS:
        c = [r for r in rows if r['cv']==cv and r['eta']==0.1]
        if not c: continue
        print(f'{cv:>5.2f} | '
              f'{msd([x["shock"] for x in c]):>11} | '
              f'{msd([x["rationing"] for x in c]):>11} | '
              f'{msd([x["repay"] for x in c]):>9} | '
              f'{msd([x["contagion"] for x in c]):>11} | '
              f'{msd([x["fiscal_deaths"] for x in c]):>11} | '
              f'{msd([x["zombies"] for x in c]):>11} | '
              f'{msd([x["bailout_bill"] for x in c]):>9} | '
              f'{msd([x["max_ten"] for x in c]):>11} | '
              f'{msd([x["avg_cli"] for x in c]):>11}')

    # Decision-tree pre-summary: which (cv, eta) cells score on all four criteria?
    print()
    print(f'=== Phase 0.5 decision criteria scan (looking for: claim3 dip + fiscal>0 + max_ten 15-50 + std/mean<50%) ===')
    print(f'{"cv":>5} | {"eta":>5} | {"bk":>7} | {"d_vs_eta0":>10} | {"fisc":>6} | {"max_ten_msd":>14} | {"std_pct":>8} | {"flags":>30}')
    print('-' * 145)
    for cv in CVS:
        bk0_cells = [r['total_bk'] for r in rows if r['cv']==cv and r['eta']==0.0]
        bk0_mean = sum(bk0_cells)/len(bk0_cells) if bk0_cells else 0
        for eta in ETAS:
            if eta == 0.0: continue
            c = [r for r in rows if r['cv']==cv and r['eta']==eta]
            if not c: continue
            bk_mean = sum(x['total_bk'] for x in c)/len(c)
            d = bk_mean - bk0_mean
            fisc = sum(x['fiscal_deaths'] for x in c)/len(c)
            mt_list = [x['max_ten'] for x in c]
            mt_mean = sum(mt_list)/len(mt_list)
            mt_std = math.sqrt(sum((x-mt_mean)**2 for x in mt_list)/max(1, len(mt_list)-1)) if len(mt_list)>1 else 0
            std_pct = (mt_std/mt_mean*100) if mt_mean > 0 else 0
            flags = []
            if d < 0: flags.append('claim3')
            if fisc > 0: flags.append('fisc>0')
            if 15 <= mt_mean <= 80: flags.append('hub_ok')
            if std_pct < 50: flags.append('low_noise')
            score = len(flags)
            highlight = ' <<<' if score == 4 else ''
            print(f'{cv:>5.2f} | {eta:>5.1f} | {bk_mean:>7.0f} | {d:>+10.0f} | {fisc:>6.0f} | '
                  f'{mt_mean:>5.1f}+/-{mt_std:>5.1f} | {std_pct:>7.1f}% | {",".join(flags):>30}{highlight}')


if __name__ == '__main__':
    main()
