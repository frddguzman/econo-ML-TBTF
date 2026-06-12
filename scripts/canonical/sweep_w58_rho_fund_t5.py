"""rho-sweep at canonical w58 cv=0, FUND REGIME ONLY at canonical tau=1e-5.

Why a separate sweep: existing sweep_w58_rho_raw.csv has fund-regime cells at
default tau=1e-4 (make_config default). The canonical commit is tau=1e-5, so
to make claim 2 multi-regime AND canonical-consistent we need the fund cells
re-run at tau=1e-5. nt and st are tau-invariant (no fund), so they stay.

Spec:
  - cand: w58 cv=0 (mu=0.7, omega=0.58, gamma=0.10, no hetero)
  - rho: {0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.70, 0.90}
  - eta: {0.0, 0.10, 0.85}
  - regime: resolution_fund only, fund_levy_rate=1e-5 (canonical)
  - seeds: 5 {26462-26466}
  - Total: 9 x 3 x 5 = 135 sims, ~12 min wall

Output: sweep_w58_rho_fund_t5_raw.csv
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
RHOS = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.70, 0.90]
ETAS = [0.0, 0.10, 0.85]
REGIME = 'resolution_fund'
TAU = 1e-5
WORKERS = 6

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_w58_rho_fund_t5_raw.csv')

KEYS = ['cand', 'mu', 'omega', 'gamma', 'rho', 'eta', 'seed', 'fiscal_regime', 'fund_levy_rate',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def run_one(args):
    rho, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=eta, regime=REGIME)
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    cfg['rho'] = rho
    cfg['fund_levy_rate'] = TAU
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
        'cand': 'w58', 'mu': 0.70, 'omega': 0.58, 'gamma': 0.10, 'rho': rho,
        'eta': eta, 'seed': seed, 'fiscal_regime': REGIME, 'fund_levy_rate': TAU,
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
    jobs = [(rho, e, s) for rho in RHOS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'rho-sweep at w58 cv=0, fund regime, tau=1e-5 (canonical)', flush=True)
    print(f'  rho: {RHOS}', flush=True)
    print(f'  eta: {ETAS}', flush=True)
    print(f'  regime: {REGIME} (tau={TAU})', flush=True)
    print(f'  seeds: {SEEDS}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers', flush=True)
    print()

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == n:
                with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=KEYS)
                    w.writeheader()
                    for rr in rows:
                        w.writerow({k: rr.get(k, '') for k in KEYS})
                print(f'  [checkpoint @ {i}/{n}]', flush=True)
    print(f'\nDONE. {n} sims written to {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
