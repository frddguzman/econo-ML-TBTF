"""Phase 0.5 expansion: 10 additional seeds (26467-26476) at the same cells.

Same config as sweep_w58_cv_phase05.py but with seeds 26467-26476 only.
Existing 5 seeds (26462-26466) are already in sweep_w58_cv_phase05_5seed.csv.
After this finishes, merge into a 15-seed CSV.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

# Extra seeds only — first 5 already done in sweep_w58_cv_phase05_5seed.csv
SEEDS  = [26467, 26468, 26469, 26470, 26471, 26472, 26473, 26474, 26475, 26476]
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
        'repay':          _sum_int(s, 'bankruptcies_repay', T),
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


def main():
    jobs = [(c, e, s) for c in CVS for e in ETAS for s in SEEDS]
    print(f'Phase 0.5 extra seeds: {len(SEEDS)} new seeds (26467-26476)')
    print(f'cv values: {CVS}')
    print(f'eta values: {ETAS}')
    print(f'{len(jobs)} sims, {WORKERS} workers')
    rows = []
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_w58_cv_phase05_extra_raw.csv')
    keys = ['cv', 'eta', 'seed', 'total_bk', 'shock', 'rationing', 'repay',
            'contagion', 'fiscal_deaths', 'zombies', 'bailout_bill',
            'avg_cli', 'avg_fitness', 'avg_ten', 'max_ten', 'turnovers']
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f'  saved -> {out_csv}')
    print('Done. Run merge_w58_cv_phase05.py to combine 5+10 seeds and produce summary.')


if __name__ == '__main__':
    main()
