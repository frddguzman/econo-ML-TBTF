"""Dashboard dump for the kept cands at the new intermediate eta=0.5.

Dashboard plan (per user 2026-05-08): keep only bsl, w55, w58 (canonical), w58_t5
as comparators. Add eta=0.5 to the existing eta dropdown {0, 0.1, 0.85}.

This script produces the missing eta=0.5 dump cells:
  - bsl_{nt,st,rf}_e05
  - w55_{nt,st,rf}_e05
  - w58_{nt,st,rf}_e05
  - w58_t5_{nt,st,rf}_e05  (note: t5 only differs from w58 in resolution_fund regime)

Total: 4 cands x 3 regimes = 12 sims at SEED=26474.

Mirror of dump_w58_cells.py with custom eta and tau handling for t5 variant.
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
ETA = 0.50
ETA_TAG = 'e05'

# (cand_tag, omega, gamma, tau_fund_override)
VARIANTS = [
    ('bsl',     0.50, 0.10, None),     # default tau=1e-4
    ('w55',     0.55, 0.10, None),
    ('w58',     0.58, 0.10, None),
    ('w58_t5',  0.58, 0.10, 1e-5),     # tau=1e-5 fund-regime canonical
]


def dump_one(args):
    cand_tag, omega, gamma, tau_override, regime_tag, regime = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=ETA, regime=regime)
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = gamma
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
    tag = f'{cand_tag}_{regime_tag}_{ETA_TAG}'
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
    for cand_tag, omega, gamma, tau in VARIANTS:
        for r_tag, regime in ddr.REGIME_DEFS:
            jobs.append((cand_tag, omega, gamma, tau, r_tag, regime))
    workers = max(1, min(len(jobs), 6))
    print(f'Dumping eta=0.5 cells for kept cands: {len(jobs)} sims, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, T, lmax, n_bd in pool.map(dump_one, jobs):
            print(f'  {tag}: T={T} lender_max={lmax} bank_rows={n_bd}')
    print(f'\nDone. {len(jobs)} eta=0.5 cells written to Simulations/.')


if __name__ == '__main__':
    main()
