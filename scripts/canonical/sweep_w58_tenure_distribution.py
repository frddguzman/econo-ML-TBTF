"""Tenure distribution sweep at w58 cv=0 — count run-lengths above multiple thresholds.

Captures the FULL run-length distribution of hub tenures, not just max_ten.
Reveals whether high max_ten values reflect sustained stability (many long runs)
or single outliers (one long run + many short ones).

Spec:
  - cand: w58 cv=0 (canonical)
  - 5 (regime, tau): nt, st, rf at tau in {1e-4, 1e-5, 1e-6}
  - 4 etas: {0.0, 0.10, 0.50, 0.85}
  - 5 seeds: {26462-26466}
  - Total: 5 x 4 x 5 = 100 sims, ~9 min wall

Per cell, computes count of runs exceeding thresholds {5, 10, 15, 20, 30, 50, 100}.
Output: sweep_w58_tenure_dist_raw.csv
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
ETAS = [0.0, 0.10, 0.50, 0.85]
WORKERS = 6
THRESHOLDS = [5, 10, 15, 20, 30, 50, 100]

REGIMES = [
    ('none',            1e-4, 'no_tax'),
    ('socialized_tax',  1e-4, 'soc_tax'),
    ('resolution_fund', 1e-4, 'rf_t1e-4'),
    ('resolution_fund', 1e-5, 'rf_t1e-5'),
    ('resolution_fund', 1e-6, 'rf_t1e-6'),
]

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_w58_tenure_dist_raw.csv')

KEYS = (['regime_label', 'fiscal_regime', 'fund_levy_rate', 'eta', 'seed',
         'n_runs', 'max_ten', 'avg_ten', 'med_ten']
        + [f'n_runs_gt_{t}' for t in THRESHOLDS]
        + [f'frac_runs_gt_{t}' for t in THRESHOLDS]
        + [f'time_gt_{t}' for t in THRESHOLDS])


def runs_from_seq(bl, T):
    raw = [int(b) for b in bl[:T] if b is not None and b >= 0]
    runs = []
    if raw:
        prev = raw[0]; rl = 1
        for k in raw[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    return runs


def run_one(args):
    regime, tau, label, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=eta, regime=regime)
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
    runs = runs_from_seq(list(m.statistics.best_lender[:T]), T)
    n_runs = len(runs)
    max_ten = max(runs) if runs else 0
    avg_ten = sum(runs)/n_runs if n_runs else 0
    med_ten = sorted(runs)[n_runs//2] if n_runs else 0
    out = {
        'regime_label': label, 'fiscal_regime': regime, 'fund_levy_rate': tau,
        'eta': eta, 'seed': seed,
        'n_runs': n_runs, 'max_ten': max_ten,
        'avg_ten': round(avg_ten, 2), 'med_ten': med_ten,
    }
    for t in THRESHOLDS:
        long_runs = [r for r in runs if r >= t]
        out[f'n_runs_gt_{t}'] = len(long_runs)
        out[f'frac_runs_gt_{t}'] = round(len(long_runs)/n_runs * 100, 2) if n_runs else 0
        out[f'time_gt_{t}'] = sum(long_runs)  # total periods spent in runs >= threshold
    return out


def main():
    jobs = [(r, t, label, e, s) for (r, t, label) in REGIMES for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'Tenure distribution sweep at w58 cv=0', flush=True)
    print(f'  5 regime variants x 4 etas x 5 seeds = {n} sims, {WORKERS} workers', flush=True)
    print()
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 25 == 0 or i == n:
                with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=KEYS)
                    w.writeheader()
                    for rr in rows:
                        w.writerow({k: rr.get(k, '') for k in KEYS})
                print(f'  [checkpoint @ {i}/{n}]', flush=True)
    print(f'\nDONE. {n} sims written to {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
