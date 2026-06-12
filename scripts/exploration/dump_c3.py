"""Dump 9 dashboard cells for the C3 cand (mu=0.60, omega=0.70).

3 regimes (nt, st, rf) x 3 etas (e0, e01, e085) = 9 cells.
Same per-period + bank-detail schema as dump_dashboard_runs.dump_one27.

Output:
  Simulations/dash_c3_<regime>_<eta>.csv         (per-period stats)
  Simulations/bank_detail_c3_<regime>_<eta>.csv  (per-bank-per-period)

Seed 26474 for visual consistency with other dashboard cands. Multi-seed
robustness already covered by thesis_lehman_c3.csv (5 seeds, no 26474).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474
MU = 0.60
OMEGA = 0.70
REGIMES = ddr.REGIME_DEFS   # [('nt','none'), ('st','socialized_tax'), ('rf','resolution_fund')]
ETAS    = ddr.ETA_DEFS      # [('e0', 0.0), ('e01', 0.1), ('e085', 0.85)]


def dump_one_c3(args):
    """Mirrors dump_one27 logic but with C3 params (mu, omega)."""
    tag, eta, regime, seed = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime=regime)
    cfg['mu'] = MU
    model = interbank.Model()
    model.test = True
    model.configure(**cfg)
    model.config.lender_change = lc.determine_algorithm("Boltzmann")
    model.initialize(seed=seed, generate_plots=False)
    model.simulate_full()
    model.finish()
    T = model.t
    stats = model.statistics

    # Per-period CSV
    columns = {'time': list(range(T))}
    empty_cols = []
    for name in ddr.STAT_NAMES_EXT:
        col = ddr.extract_array(stats, name, T)
        columns[name] = col
        if all(v is None or v == 0 for v in col):
            empty_cols.append(name)

    header = ['time'] + ddr.STAT_NAMES_EXT
    rows = [','.join(header)]
    for t in range(T):
        rows.append(','.join([str(t)] + [ddr.safe_val(columns[name][t])
                                          for name in ddr.STAT_NAMES_EXT]))
    csv_path = os.path.join(ddr.SAVE_DIR, f'dash_{tag}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))

    # Bank-detail CSV
    bd_rows = getattr(stats, '_bank_rows', []) or []
    bd_lines = [','.join(ddr.BANK_DETAIL_COLS)]
    for r in bd_rows:
        bd_lines.append(','.join(ddr.safe_val(r.get(c)) for c in ddr.BANK_DETAIL_COLS))
    bd_path = os.path.join(ddr.SAVE_DIR, f'bank_detail_{tag}.csv')
    with open(bd_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bd_lines))

    lender_max = ddr.compute_lender_max(stats, T)
    return tag, T, lender_max, len(bd_rows)


def main():
    jobs = []
    for r_tag, regime in REGIMES:
        for e_tag, eta in ETAS:
            tag = f'c3_{r_tag}_{e_tag}'
            jobs.append((tag, eta, regime, SEED))
    workers = max(1, min(len(jobs), 6))
    print(f'C3 dump (mu={MU}, omega={OMEGA}, seed={SEED}): {len(jobs)} cells, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, T, lmax, n_bd in pool.map(dump_one_c3, jobs):
            print(f'  {tag}: T={T}  lender_max={lmax}  bank_rows={n_bd}')
    print('\nDone. Add c3 to CAND_DEFS and regenerate dashboard.')


if __name__ == '__main__':
    main()
