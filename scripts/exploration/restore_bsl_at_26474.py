"""Restore bsl dashboard cells at SEED=26474 (overwritten by mistake).

bsl x 3 regimes x 4 etas {0, 0.1, 0.5, 0.85} = 12 sims at SEED=26474.
Output: ../Simulations/dash_bsl_<regime>_<eta>.csv + bank_detail_bsl_<regime>_<eta>.csv
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
ETAS = [('e0', 0.00), ('e01', 0.10), ('e05', 0.50), ('e085', 0.85)]


def dump_one(args):
    regime_tag, regime, eta_tag, eta = args
    cfg = ddr.make_config(basis='equity', omega=0.50, eta=eta, regime=regime)
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
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
        rows.append(','.join([str(t)] + [ddr.safe_val(columns[name][t]) for name in ddr.STAT_NAMES_EXT]))
    tag = f'bsl_{regime_tag}_{eta_tag}'
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
    return tag


def main():
    jobs = [(rt, r, et, e) for rt, r in ddr.REGIME_DEFS for et, e in ETAS]
    print(f'Restoring bsl cells at SEED={SEED}: {len(jobs)} sims', flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for tag in pool.map(dump_one, jobs):
            print(f'  restored: {tag}', flush=True)
    print('Done.')


if __name__ == '__main__':
    main()
