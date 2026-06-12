"""D0 expansion (option A): bsl + tight truncation × cv ∈ {0.5, 1.0} × 12 etas × 5 seeds.

Closes the bsl gap from D0 (which was cv=0.7 only). Same 3 thresholds, full eta grid.

Spec:
  - cand: bsl (μ=0.7, ω=0.50, γ=0.10)
  - cv: {0.5, 1.0}
  - 3 threshold pairs: (cap=1.3, floor=0.7), (cap=1.5, floor=0.65), (cap=1.75, floor=0.6)
  - 12 etas: full grid
  - 5 seeds: {26462-26466}
  - Total: 2 × 3 × 12 × 5 = 360 sims, ~30 min wall
  - Output: sweep_bsl_tight_truncation_cv_expand_raw.csv

Schema matches sweep_bsl_tight_truncation_raw.csv for trivial concatenation.
No in-script summary print to avoid Python 3.14 format-spec issues; analysis runs externally.
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
ETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CVS = [0.5, 1.0]
WORKERS = 6

THRESHOLDS = [
    (1.30, 0.70),
    (1.50, 0.65),
    (1.75, 0.60),
]

CAND = ('bsl', 0.70, 0.50, 0.10)

CHECKPOINT_EVERY = 50
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'sweep_bsl_tight_truncation_cv_expand_raw.csv')

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


def main():
    cand, mu, omega, gamma = CAND
    jobs = [(cand, mu, omega, gamma, cv, max_f, min_f, e, s)
            for cv in CVS for (max_f, min_f) in THRESHOLDS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'D0 expansion (A): bsl tight-truncation x cv expand', flush=True)
    print(f'  cand: {cand} (mu={mu}, omega={omega}, gamma={gamma})', flush=True)
    print(f'  cvs: {CVS}', flush=True)
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
    print(f'\nDONE. {n} sims written to {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
