"""Verify the same-bank-cycling hypothesis at high η for w58 cv=1.0 + init_replace.

Re-runs at SEED=26474 across selected etas. Captures per-period (best_lender, generation).
Computes:
  - raw-id metrics (current convention)
  - composite-key metrics (counts replacement as turnover)
  - distinct hub-id count (unique bank ids ever as hub)
  - dominant-id share (% hub-time on most-frequent id)

If hypothesis holds: at high η, distinct-id count is LOW and dominant share is HIGH,
revealing replacement-cycled lock-in despite raw-id metric showing rotation-like turnovers.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WORKERS = 5


def run_one(eta):
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = 1.0
    cfg['reintroduce_with_median'] = False
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=SEED, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T])
    bg = list(s.best_lender_generation[:T])
    return {'eta': eta, 'T': T, 'best_lender': bl, 'best_lender_generation': bg}


def runs_from_seq(seq):
    """Compute run lengths from a sequence (skipping None entries; assumes caller pre-filters invalid)."""
    seq = [v for v in seq if v is not None]
    runs = []
    if not seq: return runs
    prev = seq[0]; rl = 1
    for k in seq[1:]:
        if k == prev: rl += 1
        else: runs.append(rl); rl = 1; prev = k
    runs.append(rl)
    return runs


def stats(seq):
    runs = runs_from_seq(seq)
    if not runs: return 0, 0, 0
    return max(runs), sum(runs)/len(runs), max(0, len(runs) - 1)


def stats_int_filtered(seq):
    """For raw-id sequences: filter out negatives and run stats."""
    s = [v for v in seq if v is not None and v >= 0]
    return stats(s)


def main():
    print(f'Running {len(ETAS)} sims at seed={SEED}...', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run_one, ETAS))
    print()
    print(f'w58 cv=1.0 + reintroduce_with_median=False at SEED={SEED}')
    print('='*150)
    hdr = f'{"eta":>5} | {"raw max":>7} {"raw avg":>7} {"raw turn":>8} | {"comp max":>8} {"comp avg":>8} {"comp turn":>9} | {"#distinct ids":>13} {"dom share":>10} | {"hub time%":>9}'
    print(hdr)
    print('-'*150)
    for r in sorted(results, key=lambda x: x['eta']):
        T = r['T']
        bl = r['best_lender']
        bg = r['best_lender_generation']
        # Raw-id metrics
        raw_max, raw_avg, raw_turn = stats_int_filtered(bl)
        # Composite-key metrics: zip id and generation, treat tuple as the "key"
        composite = [(int(b), int(g)) if b is not None and b >= 0 and g is not None else None
                     for b, g in zip(bl, bg)]
        comp_seq = [k for k in composite if k is not None]
        comp_max, comp_avg, comp_turn = stats(comp_seq)
        # Distinct ids that appeared as hub
        valid_ids = [int(b) for b in bl if b is not None and b >= 0]
        distinct_ids = len(set(valid_ids))
        # Dominant-id share (% of valid hub time)
        if valid_ids:
            counts = Counter(valid_ids)
            top_id, top_count = counts.most_common(1)[0]
            dom_share = top_count / len(valid_ids) * 100
        else:
            dom_share = 0
            top_id = -1
        hub_time_pct = len(valid_ids) / T * 100  # how much of run had a valid hub
        print(f'{r["eta"]:>5.2f} | {raw_max:>7d} {raw_avg:>7.1f} {raw_turn:>8d} | '
              f'{comp_max:>8d} {comp_avg:>8.1f} {comp_turn:>9d} | '
              f'{distinct_ids:>13d} {dom_share:>9.1f}% | {hub_time_pct:>8.1f}%')


if __name__ == '__main__':
    main()
