"""Full eta sweep at C3 calibration (mu=0.6, omega=0.7) across 3 regimes,
5 seeds (26462-26466, matching thesis sweep set).

Output: Simulations/thesis_lehman_c3.csv  (same schema as thesis_lehman.csv)
        per-row: seed, fiscal_regime, rho, eta, omega, basis, inertia,
                 total_bk, shock, rationing, repay, contagion, fiscal_deaths, zombies

Ready for incremental dashboard regen / further analysis.
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

# C3 calibration
MU = 0.60
OMEGA = 0.70

ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REGIMES = ['none', 'socialized_tax', 'resolution_fund']

def _sum_int(stats, name, T):
    arr = getattr(stats, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))

def run_one(args):
    seed, eta, regime = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime=regime)
    cfg['mu'] = MU
    # median replacement default; no hetero; Boltzmann
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
        'seed':           int(seed),
        'fiscal_regime':  regime,
        'rho':            0.4,
        'eta':            float(eta),
        'omega':          OMEGA,
        'mu':             MU,
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

COLUMNS = ['seed', 'fiscal_regime', 'rho', 'eta', 'omega', 'mu', 'basis', 'inertia',
           'total_bk', 'shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths', 'zombies']

def main():
    jobs = [(s, eta, regime)
            for s in SEEDS for eta in ETAS for regime in REGIMES]
    print(f'C3 eta sweep (mu={MU}, omega={OMEGA}): {len(jobs)} sims, {WORKERS} workers')
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')
    out_path = os.path.join(SAVE_DIR, 'thesis_lehman_c3.csv')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(COLUMNS) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in COLUMNS) + '\n')
    print(f'Wrote {len(rows)} rows -> {out_path}')

    # Quick aggregate: 5-seed mean per (regime, eta)
    print('\n=== 5-seed mean total_bk by (regime, eta) ===')
    print(f'{"regime":>18} | ' + ' | '.join(f'{e:>7.1f}' for e in ETAS))
    print('-' * 110)
    for regime in REGIMES:
        means = []
        for eta in ETAS:
            cell = [r['total_bk'] for r in rows if r['fiscal_regime']==regime and r['eta']==eta]
            means.append(sum(cell)/len(cell) if cell else 0)
        print(f'{regime:>18} | ' + ' | '.join(f'{m:>7.0f}' for m in means))
        # Find min and where
        min_eta = ETAS[means.index(min(means))]
        delta_to_eta0 = min(means) - means[0]
        verdict = 'PRESERVED' if delta_to_eta0 < 0 else 'BROKEN'
        print(f'                   | min at eta={min_eta} ({min(means):.0f}), eta=0 baseline {means[0]:.0f}, delta={delta_to_eta0:+.0f} -> claim3 {verdict}')

if __name__ == '__main__':
    main()
