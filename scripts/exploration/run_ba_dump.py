"""Batch dump for Preferential (Barabási-Albert) candidates.

Runs 4 base cands × 3 regimes × 3 etas = 36 sims at the chosen Preferential m
value. Output naming: dash_<cand>_ba<m>_<regime>_<eta>.csv (and bank_detail).

Usage:
    py -3.12 run_ba_dump.py --m 3      # batch 1: classic BA
    py -3.12 run_ba_dump.py --m 1      # batch 2: pure tree

Mirrors run_w55_dump.py / dump_dashboard_runs.dump_one27 — same per-period
column schema, same bank-detail schema, same seed (26474). Only difference:
m.config.lender_change is Preferential(m=M) instead of Boltzmann.

Sets OMP/MKL/OPENBLAS thread count to 1 to avoid the numpy-threading
oversubscription that bit us during the seed search.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import argparse
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474

# 4 base cands × 3 regimes × 3 etas = 36 cells per --m
CANDS = [
    ('bsl', 'equity',    0.50),
    ('a',   'equity',    0.53),
    ('b',   'bilateral', 0.52),
    ('w55', 'equity',    0.55),
]
# Reuse ddr.REGIME_DEFS and ddr.ETA_DEFS for canonical tags
REGIMES = ddr.REGIME_DEFS   # [('nt', 'none'), ('st', 'socialized_tax'), ('rf', 'resolution_fund')]
ETAS    = ddr.ETA_DEFS      # [('e0', 0.0), ('e01', 0.1), ('e085', 0.85)]


def dump_one_ba(args):
    """Worker that runs one Preferential sim and writes per-period + bank-detail CSVs.

    Same column schema as dump_one27 — borrows STAT_NAMES_EXT and
    BANK_DETAIL_COLS plus the writer logic. Only difference: lender_change
    is Preferential with the configured m.
    """
    tag, basis, omega, eta, regime, seed, m_param = args
    cfg = ddr.make_config(basis, omega, eta, regime=regime)
    model = interbank.Model()
    model.test = True
    model.configure(**cfg)
    model.config.lender_change = lc.determine_algorithm("Preferential", m=m_param)
    model.initialize(seed=seed, generate_plots=False)
    model.simulate_full()
    model.finish()
    T = model.t
    stats = model.statistics

    # Per-period CSV (mirrors dump_one27 lines 361–377)
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

    # Bank-detail CSV (mirrors dump_one27 lines 379–387)
    bd_rows = getattr(stats, '_bank_rows', []) or []
    bd_header = ','.join(ddr.BANK_DETAIL_COLS)
    bd_lines = [bd_header]
    for r in bd_rows:
        bd_lines.append(','.join(ddr.safe_val(r.get(c)) for c in ddr.BANK_DETAIL_COLS))
    bd_path = os.path.join(ddr.SAVE_DIR, f'bank_detail_{tag}.csv')
    with open(bd_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bd_lines))

    lender_max = ddr.compute_lender_max(stats, T)
    return tag, T, lender_max, len(bd_rows), empty_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help='Barabási-Albert m parameter')
    parser.add_argument('--quick', action='store_true',
                        help='Only the chip-focused (st, eta=0.1) cell × 4 cands (4 sims, ~3 min). '
                             'Use for fast hub-dynamics inspection at a new m.')
    args = parser.parse_args()
    m_param = args.m

    # Build job list: tag = <cand>_ba<m>_<regime>_<eta>
    if args.quick:
        regimes_iter = [('st', 'socialized_tax')]
        etas_iter    = [('e01', 0.1)]
    else:
        regimes_iter = REGIMES
        etas_iter    = ETAS
    jobs = []
    for cand_tag, basis, omega in CANDS:
        for r_tag, regime in regimes_iter:
            for e_tag, eta in etas_iter:
                tag = f'{cand_tag}_ba{m_param}_{r_tag}_{e_tag}'
                jobs.append((tag, basis, omega, eta, regime, SEED, m_param))

    workers = max(1, min(len(jobs), 6))
    print(f'BA dump m={m_param} (seed={SEED}): {len(jobs)} cells × 2 CSVs each '
          f'({workers} workers, OMP=1)')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, T, lmax, n_bd, empty_cols in pool.map(dump_one_ba, jobs):
            print(f'  {tag}: T={T}  lender_max={lmax}  bank_rows={n_bd}  '
                  f'empty={len(empty_cols)}')
    print(f'\nDone. Run dump_topology_ba.py next, then regenerate dashboard.')


if __name__ == '__main__':
    main()
