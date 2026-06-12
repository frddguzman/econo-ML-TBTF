"""Two-path regression check after adding tier-init to interbank.py.

5a: cv=0 byte-equal — verify the tier-init code doesn't perturb existing cv=0 results.
    Re-run bsl + st + eta=0.1 + seed=26474 with tier_init=False, equity_heterogeneity=False,
    cash_boost_random=False. Compare bankruptcy series byte-by-byte to existing
    dash_bsl_st_e01.csv. Protects all prior cv=0 results from regression.

5b: hetero path unbroken — verify the legacy lognormal-hetero code path still runs without
    exception with tier_init=False, equity_heterogeneity=True, equity_cv=0.7.
    Cheap insurance against an inadvertent typo in the new mutex check that might affect
    the existing hetero code path.
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


def run_cv0():
    cfg = ddr.make_config(basis='equity', omega=0.50, eta=0.1, regime='socialized_tax')
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=SEED, generate_plots=False)
    m.simulate_full()
    m.finish()
    return m


def run_hetero():
    cfg = ddr.make_config(basis='equity', omega=0.50, eta=0.1, regime='socialized_tax')
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = 0.7
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


def check_cv0_byte_equal():
    print('=== 5a: cv=0 byte-equal check ===')
    print('Running fresh bsl_st_e01 sim with tier_init=False, hetero=False, CB=False...')
    m = run_cv0()
    T = m.t
    fresh_bk = [int(v) if v is not None else None for v in m.statistics.bankruptcy[:T].tolist()]

    existing_path = os.path.join(SIM_DIR, 'dash_bsl_st_e01.csv')
    print(f'Loading existing series from {existing_path}')
    existing_bk = load_existing_series(existing_path, 'bankruptcy')

    n_compared = min(len(fresh_bk), len(existing_bk))
    diffs_bk = sum(1 for a, b in zip(fresh_bk[:n_compared], existing_bk[:n_compared]) if a != b)
    total_fresh = sum(v for v in fresh_bk if v is not None)
    total_existing = sum(v for v in existing_bk if v is not None)

    print(f'  fresh   total_bk = {total_fresh}')
    print(f'  existing total_bk = {total_existing}')
    print(f'  per-period diffs  = {diffs_bk}')
    if diffs_bk == 0:
        print(f'  VERDICT 5a: byte-identical, tier-init code does not perturb cv=0 OK')
        return True
    else:
        print(f'  VERDICT 5a: REGRESSION — series differ, tier-init code CHANGED cv=0 behaviour')
        diffs_shown = 0
        for i, (a, b) in enumerate(zip(fresh_bk, existing_bk)):
            if a != b and diffs_shown < 10:
                print(f'    t={i}: fresh={a}, existing={b}')
                diffs_shown += 1
        return False


def check_hetero_unbroken():
    print()
    print('=== 5b: hetero path unbroken ===')
    print('Running fresh bsl_st_e01 sim with tier_init=False, hetero=True (cv=0.7), CB=False...')
    try:
        m = run_hetero()
        T = m.t
        total = sum(int(v) for v in m.statistics.bankruptcy[:T] if v is not None)
        print(f'  T={T}, total_bk={total}')
        print(f'  VERDICT 5b: hetero path runs without exception OK')
        return True
    except Exception as e:
        print(f'  VERDICT 5b: REGRESSION — hetero path crashed: {type(e).__name__}: {e}')
        return False


def main():
    ok_a = check_cv0_byte_equal()
    ok_b = check_hetero_unbroken()
    print()
    print('=== Summary ===')
    print(f'  5a cv=0 byte-equal:    {"PASS" if ok_a else "FAIL"}')
    print(f'  5b hetero unbroken:    {"PASS" if ok_b else "FAIL"}')
    if ok_a and ok_b:
        print('  Both regression checks PASS — safe to proceed')
        sys.exit(0)
    else:
        print('  Regression FAILED — revert via cp interbank_backup_pre_tier_init.py interbank.py')
        sys.exit(1)


if __name__ == '__main__':
    main()
