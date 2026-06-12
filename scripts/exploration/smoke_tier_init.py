"""Smoke test for tier-init: balance-sheet integrity, E-magnitudes, and tier persistence.

Runs 1 sim at bsl + tier_init=True, n_big=3, E_big_multiplier=3.0, SEED=26462.
Asserts:
  1. Balance-sheet identity at t=0 for all 50 banks
  2. E-magnitudes at t=0 (3 big banks with E > 30, 47 small banks with E < 20)
  3. After full T=1000 run: tier composition preserved (count of 'big' banks == 3)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr


def main():
    cfg = ddr.make_config(basis='equity', omega=0.50, eta=0.1, regime='socialized_tax')
    cfg['tier_init'] = True
    cfg['n_big'] = 3
    cfg['E_big_multiplier'] = 3.0

    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=26462, generate_plots=False)

    # 1. Balance-sheet identity at t=0
    for b in m.banks:
        bs_diff = abs(b.A - (b.D + b.E)) if b.A else abs((b.L + b.C + b.R) - (b.D + b.E))
        assert bs_diff < 1e-6, f'bank {b.id} balance-sheet violation at t=0: |L+C+R - (D+E)| = {bs_diff}'
    # b.A is set during simulate_full; at t=0 we check L+C+R = D+E directly
    for b in m.banks:
        lhs = b.L + b.C + b.R
        rhs = b.D + b.E
        assert abs(lhs - rhs) < 1e-6, f'bank {b.id}: L+C+R={lhs}, D+E={rhs}'
    print('balance-sheet identity OK at t=0 for all 50 banks')

    # 2. E-magnitudes at t=0
    big_banks = [b for b in m.banks if b.tier == 'big']
    small_banks = [b for b in m.banks if b.tier == 'small']
    assert len(big_banks) == 3, f'expected 3 big banks at t=0, got {len(big_banks)}'
    assert all(b.E > 30 for b in big_banks), f'big banks not actually big: {[b.E for b in big_banks]}'
    assert all(b.E < 20 for b in small_banks), \
        f'small banks not actually small: {[(b.id, b.E) for b in small_banks if b.E >= 20]}'
    print('big-bank E magnitudes OK at t=0')
    print(f'  big banks (n={len(big_banks)}):   E values = {[round(b.E, 2) for b in big_banks]}')
    print(f'  small banks (n={len(small_banks)}): E mean = {sum(b.E for b in small_banks)/len(small_banks):.2f}, '
          f'min = {min(b.E for b in small_banks):.2f}, max = {max(b.E for b in small_banks):.2f}')

    # 3. Run sim and check tier composition preserved
    print('Running simulate_full() with T=1000 ...')
    m.simulate_full()
    m.finish()
    big_count_end = sum(1 for b in m.banks if b.tier == 'big')
    assert big_count_end == 3, f'tier composition violated at end: expected 3 big banks, got {big_count_end}'
    print(f'tier composition preserved: {big_count_end} big banks at end of T={m.t} run')

    # Final stats
    T = m.t
    import numpy as np
    total_bk = int(np.nansum(m.statistics.bankruptcy[:T]))
    bl = list(m.statistics.best_lender[:T])
    raw_ids = [v for v in bl if v >= 0]
    runs = []
    if raw_ids:
        prev = raw_ids[0]; rl = 1
        for k in raw_ids[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    print()
    print(f'=== Final stats ===')
    print(f'  T              = {T}')
    print(f'  total_bk       = {total_bk}')
    print(f'  max_ten        = {max_ten}')
    print(f'  big tier IDs at end: {sorted(b.id for b in m.banks if b.tier == "big")}')
    print()
    print('SMOKE TEST: ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
