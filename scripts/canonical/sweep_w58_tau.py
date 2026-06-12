"""tau_fund tuning at w58 (mu=0.7, omega=0.58, gamma=0.10) under fund regime.

Tests user's hypothesis: w58 fund regime is monotonic-up not because the model
is fundamentally broken at higher omega, but because tau_fund=0.0001 (calibrated
for bsl-magnitude bk ~17k) is too small relative to w58's bk ~3k per-period
bailout demand at moderate eta.

Sweep tau ∈ {0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002} (5x below to 20x above default).

Track everything: bankruptcy channels (shock/rationing/repay/contagion/fiscal/zombies),
hub stats (raw-id max_ten, avg_ten, turnovers, avg_cli, avg hub fitness),
and total_bk for claim 3 verdict.

Cells: 6 tau x 7 eta x 5 seeds x fund regime = 210 sims, ~17 min.
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
REGIME = 'resolution_fund'

TAUS = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _sum_int(stats_obj, name, T):
    arr = getattr(stats_obj, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    tau, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime=REGIME)
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['fund_levy_rate'] = tau
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

    return {
        'tau': tau, 'eta': eta, 'seed': seed,
        'total_bk':       _sum_int(s, 'bankruptcy', T),
        'shock':          _sum_int(s, 'bankruptcies_shock', T),
        'rationing':      _sum_int(s, 'bankruptcies_rationing', T),
        'repay':          _sum_int(s, 'bankruptcies_repay', T),
        'contagion':      _sum_int(s, 'bankruptcies_contagion', T),
        'fiscal_deaths':  _sum_int(s, 'bankruptcies_fiscal', T),
        'zombies':        _sum_int(s, 'fire_sale_survivors', T),
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
    jobs = [(t, e, s) for t in TAUS for e in ETAS for s in SEEDS]
    print(f'tau_fund tuning at w58 (mu={MU}, omega={OMEGA}, gamma={GAMMA}, fund regime)')
    print(f'{len(jobs)} sims, {WORKERS} workers')
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_w58_tau_raw.csv')
    keys = ['tau', 'eta', 'seed', 'total_bk', 'shock', 'rationing', 'repay',
            'contagion', 'fiscal_deaths', 'zombies',
            'avg_cli', 'avg_fitness', 'avg_ten', 'max_ten', 'turnovers']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f'  saved raw rows -> {out_csv}')

    # Per-tau eta-curve for total_bk (claim 3 verdict)
    print()
    print(f'=== Total bk by tau and eta (5-seed mean) ===')
    print(f'{"tau":>8} | ' + ' | '.join(f'{"eta=" + str(e):>9}' for e in ETAS) + ' | claim3 verdict')
    print('-' * 130)
    for tau in TAUS:
        means = []
        for eta in ETAS:
            cell = [r['total_bk'] for r in rows if r['tau']==tau and r['eta']==eta]
            means.append(sum(cell)/len(cell) if cell else 0)
        min_eta = ETAS[means.index(min(means))]
        d = min(means) - means[0]
        verdict = f'YES (eta*={min_eta}, delta={d:+.0f})' if d < 0 else f'NO (mono up)'
        print(f'{tau:>8.5f} | ' + ' | '.join(f'{m:>9.0f}' for m in means) + f' | {verdict}')

    # Decomposition table at eta=0.1 (the original etastar) per tau
    print()
    print(f'=== Channel decomposition at eta=0.1 fund regime (5-seed mean +/- std) ===')
    print(f'{"tau":>8} | {"shock":>11} | {"rationing":>11} | {"repay":>9} | '
          f'{"contagion":>11} | {"fiscal":>9} | {"zombies":>11} | '
          f'{"max_ten":>9} | {"avg_cli":>9} | {"hub_fit":>9}')
    print('-' * 145)
    for tau in TAUS:
        c = [r for r in rows if r['tau']==tau and r['eta']==0.1]
        if not c: continue
        print(f'{tau:>8.5f} | '
              f'{msd([x["shock"] for x in c]):>11} | '
              f'{msd([x["rationing"] for x in c]):>11} | '
              f'{msd([x["repay"] for x in c]):>9} | '
              f'{msd([x["contagion"] for x in c]):>11} | '
              f'{msd([x["fiscal_deaths"] for x in c]):>9} | '
              f'{msd([x["zombies"] for x in c]):>11} | '
              f'{msd([x["max_ten"] for x in c]):>9} | '
              f'{msd([x["avg_cli"] for x in c]):>9} | '
              f'{msd([x["avg_fitness"] for x in c]):>9}')


if __name__ == '__main__':
    main()
