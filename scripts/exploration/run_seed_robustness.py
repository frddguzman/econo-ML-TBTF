"""Multi-seed lender-tenure robustness check at the chip-focused cell (η=0.1, ρ=0.4).

For each (cand, regime, seed) triple, run a single sim and compute lender max
and mean tenure (RLE on (best_lender, best_lender_generation)). Output a flat
CSV the dashboard can group/aggregate.

Goal: settle the "max=24 vs 8" seed-overfit concern. Across 30 seeds, do the
3 fiscal regimes show statistically distinguishable tenure distributions, or
is seed 26474's max=24 socialized_tax just a one-off?

Run: py -3.12 run_seed_robustness.py
Output: Simulations/seed_robustness_eta01.csv
Total: 4 cands × 3 regimes × 30 seeds = 360 sims, ~5 min on 11 cores at T=1000.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_dashboard_runs as ddr

SAVE_DIR = ddr.SAVE_DIR
CANDS = [
    ('bsl', 'equity',    0.50),
    ('a',   'equity',    0.53),
    ('b',   'bilateral', 0.52),
    ('w55', 'equity',    0.55),
]
REGIMES = [
    ('nt', 'none'),
    ('st', 'socialized_tax'),
    ('rf', 'resolution_fund'),
]
SEEDS = list(range(26462, 26492))   # 30 seeds
ETA = 0.1


def _rle_runs(keys):
    if not keys: return []
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return runs


def run_one(args):
    cand_tag, basis, omega, regime_tag, regime, seed = args
    m = ddr.run_sim(basis, omega, ETA, seed, regime=regime)
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T])
    bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    runs = _rle_runs(keys)
    max_tenure = max(runs) if runs else 0
    avg_tenure = (sum(runs) / len(runs)) if runs else 0.0
    total_bk = int(sum(s.bankruptcy[:T]))
    return {
        'seed':        int(seed),
        'cand':        cand_tag,
        'regime':      regime_tag,
        'max_tenure':  int(max_tenure),
        'avg_tenure':  round(float(avg_tenure), 3),
        'total_bk':    total_bk,
    }


def main():
    jobs = []
    for cand_tag, basis, omega in CANDS:
        for regime_tag, regime in REGIMES:
            for seed in SEEDS:
                jobs.append((cand_tag, basis, omega, regime_tag, regime, seed))
    workers = max(1, min(len(jobs), 6))   # 6 physical cores; cap to avoid oversubscription
    print(f'seed-robustness sweep: {len(jobs)} sims at eta={ETA}, '
          f'{workers} workers')
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(run_one, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')
    out_path = os.path.join(SAVE_DIR, 'seed_robustness_eta01.csv')
    cols = ['seed', 'cand', 'regime', 'max_tenure', 'avg_tenure', 'total_bk']
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'\nDone. Wrote {len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()
