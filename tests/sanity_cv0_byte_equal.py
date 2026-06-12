"""Verify that the np.random fix doesn't perturb cv=0 outputs.

Strategy: re-run bsl + st + eta=0.1 + seed=26474 (the canonical Phase 0 dump cell),
compare the per-period bankruptcy series to the existing dash_bsl_st_e01.csv that
was generated before the fix. Should be byte-identical.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474
SIM_DIR = ddr.SAVE_DIR


def run_one():
    # Mirror dump_dashboard_runs.run_sim exactly
    cfg = ddr.make_config(basis='equity', omega=0.50, eta=0.1, regime='socialized_tax')
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=SEED, generate_plots=False)
    m.simulate_full()
    m.finish()
    return m


def load_existing_series(path, col):
    out = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            v = r.get(col, '')
            try:
                out.append(int(float(v)))
            except (TypeError, ValueError):
                out.append(None)
    return out


def main():
    print('Running fresh bsl_st_e01 sim with the fix...')
    m = run_one()
    T = m.t
    fresh_bk = [int(v) if v is not None else None for v in m.statistics.bankruptcy[:T].tolist()]
    fresh_eq = [round(float(v), 6) if v is not None else None for v in m.statistics.equity[:T].tolist()]

    existing_path = os.path.join(SIM_DIR, 'dash_bsl_st_e01.csv')
    print(f'Loading existing series from {existing_path}')
    existing_bk = load_existing_series(existing_path, 'bankruptcy')

    n_compared = min(len(fresh_bk), len(existing_bk))
    diffs_bk = sum(1 for a, b in zip(fresh_bk[:n_compared], existing_bk[:n_compared]) if a != b)
    total_fresh = sum(v for v in fresh_bk if v is not None)
    total_existing = sum(v for v in existing_bk if v is not None)

    print()
    print(f'=== bankruptcy series comparison (T={n_compared}) ===')
    print(f'  fresh   total_bk = {total_fresh}')
    print(f'  existing total_bk = {total_existing}')
    print(f'  per-period diffs  = {diffs_bk}')
    if diffs_bk == 0:
        print(f'  VERDICT: byte-identical, fix does not perturb cv=0 ✓')
    else:
        print(f'  VERDICT: ⚠️  series differ — fix CHANGED cv=0 behaviour')
        # Show first 10 diff positions
        diffs_shown = 0
        for i, (a, b) in enumerate(zip(fresh_bk, existing_bk)):
            if a != b and diffs_shown < 10:
                print(f'    t={i}: fresh={a}, existing={b}')
                diffs_shown += 1


if __name__ == '__main__':
    main()
