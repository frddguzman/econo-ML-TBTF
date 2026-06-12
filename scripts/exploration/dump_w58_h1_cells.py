"""Dashboard dump for w58_h1 family: w58 cv=1.0 + reintroduce_with_median=False.

Adds a new heterogeneous cand to the dashboard at all 3 regimes x 4 etas {0, 0.1, 0.5, 0.85}.

Cand naming convention follows w58_t5/w58_t6 pattern:
  - w58_h1: hetero (cv=1.0) + init_replace, default tau=1e-4
  - w58_h1_t5: same hetero, tau=1e-5 (matches canonical fund-regime tau)
  - w58_h1_t6: same hetero, tau=1e-6

Note: nt and st regimes don't use tau, so the t5/t6 variants have IDENTICAL nt/st
cells to default w58_h1. Only rf differs across tau. To keep dashboard structure
consistent (full 9-cell-per-cand grid), we dump all 12 cells per cand anyway.

Total: 3 cands x 3 regimes x 4 etas = 36 sims at SEED=26474.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.stdout.reconfigure(encoding='utf-8')
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474
OMEGA = 0.58
GAMMA = 0.10
CV = 1.0

# 4 etas including the new e05 intermediate
ETAS = [
    ('e0',   0.00),
    ('e01',  0.10),
    ('e05',  0.50),
    ('e085', 0.85),
]

# (cand_tag, tau_fund_override)
VARIANTS = [
    ('w58_h1',    None),    # default tau=1e-4
    ('w58_h1_t5', 1e-5),    # tau=1e-5 (canonical fund-regime tau)
    ('w58_h1_t6', 1e-6),
]


def dump_one(args):
    cand_tag, tau_override, regime_tag, regime, eta_tag, eta = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime=regime)
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = GAMMA
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = CV
    cfg['reintroduce_with_median'] = False  # the D1 axis
    if tau_override is not None:
        cfg['fund_levy_rate'] = tau_override
    model = interbank.Model()
    model.test = True
    model.configure(**cfg)
    model.config.lender_change = lc.determine_algorithm("Boltzmann")
    model.initialize(seed=SEED, generate_plots=False)
    model.simulate_full()
    model.finish()
    T = model.t
    stats = model.statistics

    columns = {'time': list(range(T))}
    for name in ddr.STAT_NAMES_EXT:
        columns[name] = ddr.extract_array(stats, name, T)
    header = ['time'] + ddr.STAT_NAMES_EXT
    rows = [','.join(header)]
    for t in range(T):
        rows.append(','.join([str(t)] + [ddr.safe_val(columns[name][t])
                                          for name in ddr.STAT_NAMES_EXT]))
    tag = f'{cand_tag}_{regime_tag}_{eta_tag}'
    csv_path = os.path.join(ddr.SAVE_DIR, f'dash_{tag}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))

    bd_rows = getattr(stats, '_bank_rows', []) or []
    bd_lines = [','.join(ddr.BANK_DETAIL_COLS)]
    for r in bd_rows:
        bd_lines.append(','.join(ddr.safe_val(r.get(c)) for c in ddr.BANK_DETAIL_COLS))
    bd_path = os.path.join(ddr.SAVE_DIR, f'bank_detail_{tag}.csv')
    with open(bd_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bd_lines))

    return tag, T, ddr.compute_lender_max(stats, T), len(bd_rows)


def main():
    jobs = []
    for cand_tag, tau in VARIANTS:
        for r_tag, regime in ddr.REGIME_DEFS:
            for e_tag, eta in ETAS:
                jobs.append((cand_tag, tau, r_tag, regime, e_tag, eta))
    workers = max(1, min(len(jobs), 6))
    print(f'Dumping w58_h1 family cells: {len(jobs)} sims, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, T, lmax, n_bd in pool.map(dump_one, jobs):
            print(f'  {tag}: T={T} lender_max={lmax} bank_rows={n_bd}')
    print(f'\nDone. {len(jobs)} w58_h1-family cells written to Simulations/.')


if __name__ == '__main__':
    main()
