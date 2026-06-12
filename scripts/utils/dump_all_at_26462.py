"""Dump dashboard cells at SEED=26462 with _s62 suffix (alongside existing 26474 cells).

Adds new cands to the dashboard with _s62 suffix in tags so both seed=26474 (original)
and seed=26462 (thesis median) versions coexist for side-by-side comparison.

Cells dumped (with _s62 suffix in cand tag):
  4 kept cands (bsl_s62, w55_s62, w58_s62, w58_t5_s62) x 3 regimes x 4 etas {0, 0.1, 0.5, 0.85}
+ 3 hetero cands (w58_h1_s62, w58_h1_t5_s62, w58_h1_t6_s62) x 3 regimes x 4 etas
= 84 sims total at SEED=26462, ~7-10 min wall.

Output: ../Simulations/dash_<cand>_s62_<regime>_<eta>.csv + bank_detail_*.csv (NEW files, no overwrite).
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

SEED = 26462
ETAS = [
    ('e0',   0.00),
    ('e01',  0.10),
    ('e05',  0.50),
    ('e085', 0.85),
]

# (cand_tag with _s62 suffix, omega, gamma, equity_hetero, cv, init_replace, tau)
VARIANTS = [
    # Kept cands at SEED=26462 (thesis median seed)
    ('bsl_s62',         0.50, 0.10, False, 0.0, True,  None),
    ('w55_s62',         0.55, 0.10, False, 0.0, True,  None),
    ('w58_s62',         0.58, 0.10, False, 0.0, True,  None),
    ('w58_t5_s62',      0.58, 0.10, False, 0.0, True,  1e-5),
    # Heterogeneous variants at SEED=26462
    ('w58_h1_s62',      0.58, 0.10, True,  1.0, False, None),
    ('w58_h1_t5_s62',   0.58, 0.10, True,  1.0, False, 1e-5),
    ('w58_h1_t6_s62',   0.58, 0.10, True,  1.0, False, 1e-6),
]


def dump_one(args):
    cand_tag, omega, gamma, hetero, cv, reintr_med, tau, regime_tag, regime, eta_tag, eta = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime=regime)
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = gamma
    if hetero:
        cfg['equity_heterogeneity'] = True
        cfg['equity_cv'] = cv
    cfg['reintroduce_with_median'] = reintr_med
    if tau is not None:
        cfg['fund_levy_rate'] = tau
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

    return tag, T


def main():
    jobs = []
    for v in VARIANTS:
        for r_tag, regime in ddr.REGIME_DEFS:
            for e_tag, eta in ETAS:
                jobs.append(v + (r_tag, regime, e_tag, eta))
    workers = max(1, min(len(jobs), 6))
    print(f'Dumping ALL dashboard cells at SEED={SEED}: {len(jobs)} sims, {workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (tag, T) in enumerate(pool.map(dump_one, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f'  [{i}/{len(jobs)}] last: {tag} (T={T})', flush=True)
    print(f'\nDone. {len(jobs)} cells written to ../Simulations/ at SEED={SEED}.')


if __name__ == '__main__':
    main()
