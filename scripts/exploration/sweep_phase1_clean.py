"""Phase 1: full thesis sweeps for 4 clean (cv=0) cells.

Cells:
  w58       (mu=0.7, omega=0.58, gamma=0.10) - cleanest signal (12% std/mean)
  c3_g05    (mu=0.6, omega=0.70, gamma=0.05) - best hub of C3 family
  c3_g12    (mu=0.6, omega=0.70, gamma=0.12) - deepest claim 3 of C3 family
  w58_g04   (mu=0.7, omega=0.58, gamma=0.04) - max hub at w58, marginal claim 3

Each cell: 5 seeds x 10 etas x 3 regimes = 150 sims.
Total: 600 sims, ~50 min wall.

Output: thesis_lehman_<tag>.csv per cell, schema-compatible with existing
thesis_lehman_*.csv files for dashboard consumption.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Simulations')
SEEDS    = [26462, 26463, 26464, 26465, 26466]
WORKERS  = 6

ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REGIMES = ['none', 'socialized_tax', 'resolution_fund']

# (tag, mu, omega, gamma) - cv=0 (no hetero) for all
CELLS = [
    ('w58',     0.70, 0.58, 0.10),
    ('c3_g05',  0.60, 0.70, 0.05),
    ('c3_g12',  0.60, 0.70, 0.12),
    ('w58_g04', 0.70, 0.58, 0.04),
]


def _sum_int(stats_obj, name, T):
    arr = getattr(stats_obj, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    tag, mu, omega, gamma, seed, eta, regime = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime=regime)
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    return {
        'tag':            tag,
        'seed':           int(seed),
        'fiscal_regime':  regime,
        'rho':            0.4,
        'eta':            float(eta),
        'omega':          omega,
        'mu':             mu,
        'gamma':          gamma,
        'basis':          'equity',
        'inertia':        0.0,
        'total_bk':       _sum_int(s, 'bankruptcy', T),
        'shock':          _sum_int(s, 'bankruptcies_shock', T),
        'rationing':      _sum_int(s, 'bankruptcies_rationing', T),
        'repay':          _sum_int(s, 'bankruptcies_repay', T),
        'contagion':      _sum_int(s, 'bankruptcies_contagion', T),
        'fiscal_deaths':  _sum_int(s, 'bankruptcies_fiscal', T),
        'zombies':        _sum_int(s, 'fire_sale_survivors', T),
    }


COLUMNS = ['seed', 'fiscal_regime', 'rho', 'eta', 'omega', 'mu', 'gamma', 'basis', 'inertia',
           'total_bk', 'shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths', 'zombies']


def main():
    for tag, mu, omega, gamma in CELLS:
        jobs = [(tag, mu, omega, gamma, seed, eta, regime)
                for seed in SEEDS for eta in ETAS for regime in REGIMES]
        print(f'\n=== {tag}: mu={mu}, omega={omega}, gamma={gamma} ({len(jobs)} sims) ===')
        rows = []
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for i, r in enumerate(ex.map(run_one, jobs), 1):
                rows.append(r)
                if i % 30 == 0 or i == len(jobs):
                    print(f'  {i}/{len(jobs)} done')
        out_path = os.path.join(SAVE_DIR, f'thesis_lehman_{tag}.csv')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(','.join(COLUMNS) + '\n')
            for r in rows:
                f.write(','.join(str(r[c]) for c in COLUMNS) + '\n')
        print(f'  wrote {len(rows)} rows -> {out_path}')

        # Per-regime quick aggregate
        print('  --- 5-seed mean total_bk by (regime, eta) ---')
        for regime in REGIMES:
            means = []
            for eta in ETAS:
                cell = [r['total_bk'] for r in rows
                        if r['fiscal_regime']==regime and r['eta']==eta]
                means.append(sum(cell)/len(cell) if cell else 0)
            print(f'    {regime:>16}: ' + ' '.join(f'{m:>6.0f}' for m in means))
            min_eta = ETAS[means.index(min(means))]
            d = min(means) - means[0]
            verdict = 'YES' if d < 0 else 'NO'
            print(f'                     min at eta={min_eta} ({min(means):.0f}), '
                  f'eta=0 {means[0]:.0f}, delta={d:+.0f} -> claim3 {verdict}')


if __name__ == '__main__':
    main()
