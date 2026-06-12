"""Single-seed w55 dump: 9 cells (3 regimes × 3 etas) at omega=0.55, equity basis,
seed 26474. Adds a 4th 'control' candidate to the dashboard so the user can
inspect hub-tracker stabilization at omega=0.55 (the value where the omega-sweep
metric stabilizes) using the same per-period view that the 27-cell grid uses.

Output: 9 dash_w55_<regime>_<eta>.csv files in Simulations/.
        9 bank_detail_w55_<regime>_<eta>.csv files (graceful-degrade if empty).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_dashboard_runs as ddr

SEED = 26474
BASIS = 'equity'
OMEGA = 0.55
CAND_TAG = 'w55'

CONFIGS = []
for r_tag, regime in ddr.REGIME_DEFS:
    for e_tag, eta in ddr.ETA_DEFS:
        tag = f'{CAND_TAG}_{r_tag}_{e_tag}'
        CONFIGS.append((tag, BASIS, OMEGA, eta, regime, SEED))


def main():
    from concurrent.futures import ProcessPoolExecutor
    workers = max(1, min(len(CONFIGS), 6))   # 6 physical cores cap
    print(f'w55 dump (seed={SEED}, basis={BASIS}, omega={OMEGA}): '
          f'{len(CONFIGS)} cells, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, T, lmax, n_bd, empty_cols in pool.map(ddr.dump_one27, CONFIGS):
            print(f'  {tag}: T={T}  lender_max={lmax}  bank_rows={n_bd}'
                  f'  empty={len(empty_cols)}')
    print('\nDone.  Update gen_dashboard.py CAND_DEFS to include w55, then re-run gen_dashboard.py.')


if __name__ == '__main__':
    main()
