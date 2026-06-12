"""Smoking gun: intermediate τ values between 1e-5 and 1e-4.

Goal: see if an intermediate τ preserves the η* interior optimum WHILE having
more dynamics than canonical τ=1e-5 (i.e., higher fiscal_deaths / contagion / total_bk
than canonical but lower than τ=1e-4).

Spec:
- τ ∈ {3e-5, 5e-5, 7e-5}  (between 1e-5 and 1e-4)
- η ∈ {0, 0.10, 0.50, 0.85}  (claim 4 anchors + interior)
- regime: resolution_fund only (canonical class)
- 5 seeds {26462-26466}
- Total: 3 × 4 × 5 = 60 sims, ~5 min

Output: sweep_intermediate_levy_smoke_raw.csv

Stats: total_bk, contagion, rationing, repay, fiscal_deaths, zombies, max_ten,
avg_ten, avg_cli, bailout_bill (same as canonical full grid).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]
TAUS = [3e-5, 5e-5, 7e-5]
ETAS = [0.0, 0.10, 0.50, 0.85]
WORKERS = 6

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_intermediate_levy_smoke_raw.csv')

KEYS = ['cand', 'mu', 'omega', 'gamma', 'tau', 'eta', 'seed', 'fiscal_regime',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def run_one(args):
    tau, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=eta, regime='resolution_fund')
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
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
    bill = float(np.nansum(s.bailout_bill[:T])) if hasattr(s, 'bailout_bill') else 0.0
    def _sum(name):
        arr = getattr(s, name, None)
        if arr is None: return 0
        return int(np.nansum(arr[:T]))
    return {
        'cand': 'w58', 'mu': 0.70, 'omega': 0.58, 'gamma': 0.10,
        'tau': tau, 'eta': eta, 'seed': seed, 'fiscal_regime': 'resolution_fund',
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


def main():
    jobs = [(t, e, s) for t in TAUS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'intermediate-tau smoke: w58 cv=0, ex-ante regime', flush=True)
    print(f'  taus: {TAUS}', flush=True)
    print(f'  etas: {ETAS}', flush=True)
    print(f'  seeds: {SEEDS}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers', flush=True)
    print()

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 15 == 0 or i == n:
                with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=KEYS)
                    w.writeheader()
                    for rr in rows:
                        w.writerow({k: rr.get(k, '') for k in KEYS})
                print(f'  [checkpoint @ {i}/{n}]', flush=True)
    print(f'\nDONE. {n} sims written to {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
