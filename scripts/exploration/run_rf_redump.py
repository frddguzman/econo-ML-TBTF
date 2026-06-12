"""Re-dump the 9 resolution_fund cells of the 27-cell grid at the corrected
fund_levy_rate=0.0001 (gui_zombie default). Overwrites:
    Simulations/dash_<bsl|a|b>_rf_<e0|e01|e085>.csv
    Simulations/bank_detail_<bsl|a|b>_rf_<e0|e01|e085>.csv

Same seed (26474) as the original dump27. The 18 nt/st cells are unaffected
because their fiscal regimes don't use fund_levy_rate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dump_dashboard_runs as ddr

SEED = 26474

CONFIGS = []
for c_tag, basis, omega in ddr.CAND_DEFS:
    for e_tag, eta in ddr.ETA_DEFS:
        tag = f'{c_tag}_rf_{e_tag}'
        CONFIGS.append((tag, basis, omega, eta, 'resolution_fund', SEED))


def main():
    from concurrent.futures import ProcessPoolExecutor
    workers = max(1, min(len(CONFIGS), 6))   # 6 physical cores cap
    print(f'rf re-dump (seed={SEED}, fund_levy_rate=0.0001): '
          f'{len(CONFIGS)} cells, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, T, lmax, n_bd, empty_cols in pool.map(ddr.dump_one27, CONFIGS):
            print(f'  {tag}: T={T}  lender_max={lmax}  bank_rows={n_bd}'
                  f'  empty={len(empty_cols)}')
    print('\nDone.  Re-run gen_dashboard.py to embed.')


if __name__ == '__main__':
    main()
