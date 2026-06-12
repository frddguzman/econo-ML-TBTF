"""D1 full eta sweep: hetero + reintroduce_with_median=False at bsl + w58, cv ∈ {0.7, 1.0}.

Goal: characterize the transition between multi-hub rotation regime and champion regime
across full eta range. Determine if claim 3 has clean U-shape or is artifact-driven.

Spec:
  - cands: bsl (μ=0.7, ω=0.50), w58 (μ=0.7, ω=0.58)
  - cv: {0.7, 1.0}
  - etas (15 with extra resolution): {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                       0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90}
  - seeds: {26462-26466}
  - reintroduce_with_median: False
  - regime: socialized_tax
  - Total: 2 × 2 × 15 × 5 = 300 sims, ~25 min wall

Output: sweep_d1_full_eta_raw.csv (same schema as Approach B for comparison).
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
ETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
CVS = [0.7, 1.0]
WORKERS = 6

CANDS = [
    ('bsl', 0.70, 0.50, 0.10),
    ('w58', 0.70, 0.58, 0.10),
]

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_d1_full_eta_raw.csv')

KEYS = ['cand', 'mu', 'omega', 'gamma', 'cv', 'reintr_median',
        'eta', 'seed', 'fiscal_regime',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def run_one(args):
    cand, mu, omega, gamma, cv, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = cv
    cfg['reintroduce_with_median'] = False
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
        'reintr_median': False,
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
    jobs = [(c[0], c[1], c[2], c[3], cv, e, s)
            for c in CANDS for cv in CVS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'D1 full eta sweep: hetero + reintroduce_with_median=False', flush=True)
    print(f'  cands: {[c[0] for c in CANDS]}', flush=True)
    print(f'  cvs: {CVS}', flush=True)
    print(f'  etas: {ETAS}', flush=True)
    print(f'  seeds: {SEEDS}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers', flush=True)
    print()

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 50 == 0 or i == n:
                with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=KEYS)
                    w.writeheader()
                    for rr in rows:
                        w.writerow({k: rr.get(k, '') for k in KEYS})
                print(f'  [checkpoint @ {i}/{n}]', flush=True)
    print(f'\nDONE. {n} sims written to {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
