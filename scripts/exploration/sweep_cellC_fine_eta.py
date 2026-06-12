"""Cell C fine-eta sweep: bsl + cv=0.7, cap=1.75, floor=0.60 across the STAR → NO-HUB transition zone.

Existing observation:
  - η ∈ [0, 0.7]: uniform mega-star
  - η = 0.80: 4/5 stars + 1 no-hub (transition cell)
  - η = 0.90: 5/5 no-hub

Goal: characterize the transition with finer eta resolution.

Spec:
  - cand: bsl (μ=0.7, ω=0.50, γ=0.10), cv=0.7, cap=1.75, floor=0.60
  - regime: socialized_tax
  - etas: {0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90}
  - seeds: {26462-26466}
  - Total: 10 × 5 = 50 sims, ~5 min wall
  - Output: sweep_cellC_fine_eta_raw.csv

Schema matches D0 sweeps for trivial concatenation/comparison.
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
# Full eta range with finer resolution in the transition zone
ETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.65,
        0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.95]
WORKERS = 6

CV = 0.7
MAX_F = 1.75
MIN_F = 0.60
MU = 0.70
OMEGA = 0.50
GAMMA = 0.10

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'sweep_cellC_fine_eta_raw.csv')

KEYS = ['cand', 'mu', 'omega', 'gamma', 'cv', 'max_factor', 'min_factor',
        'eta', 'seed', 'fiscal_regime',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def run_one(args):
    eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime='socialized_tax')
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = CV
    cfg['equity_max_factor'] = MAX_F
    cfg['equity_min_factor'] = MIN_F
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
        'cand': 'bsl', 'mu': MU, 'omega': OMEGA, 'gamma': GAMMA, 'cv': CV,
        'max_factor': MAX_F, 'min_factor': MIN_F,
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


def main():
    jobs = [(e, s) for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'Cell C fine-eta sweep: bsl cv={CV} cap={MAX_F} floor={MIN_F}', flush=True)
    print(f'  etas: {ETAS}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers', flush=True)
    print('', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 10 == 0 or i == n:
                with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=KEYS)
                    w.writeheader()
                    for rr in rows:
                        w.writerow({k: rr.get(k, '') for k in KEYS})
                print(f'  [checkpoint @ {i}/{n}]', flush=True)
    print(f'\nDONE. {n} sims written to {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
