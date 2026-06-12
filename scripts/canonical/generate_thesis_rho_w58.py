"""Generate thesis_rho_w58*.csv from existing sweep_w58_rho_raw.csv.

Per dashboard convention (claim_2.tex, thesis_rho.csv schema):
  - thesis_rho_<cand>.csv: rho-sweep at (eta=0, regime=none) — no-tax only
  - thesis_rho01_<cand>.csv: eta-sweep at rho=0.1, regimes=[none, resolution_fund]

Output schema (matches existing thesis_rho.csv):
  seed, fiscal_regime, rho, eta, omega, basis, inertia,
  total_bk, shock, rationing, repay, contagion, fiscal_deaths, zombies

For tau variants (w58_t5, w58_t6):
  - thesis_rho_w58_t5/t6.csv: identical to thesis_rho_w58.csv (no-tax is tau-invariant)
  - thesis_rho01_w58_t5/t6.csv: COPIED from thesis_rho01_w58.csv as placeholder
    (fund-regime data at tau=1e-4, not actual tau=1e-5/1e-6 — flagged for future work).
"""
import os, csv, sys, shutil
sys.stdout.reconfigure(encoding='utf-8')

SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Simulations')
SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_w58_rho_raw.csv')

# Schema of existing thesis_rho.csv (column order matters for some tools)
THESIS_COLS = ['seed', 'fiscal_regime', 'rho', 'eta', 'omega', 'basis', 'inertia',
               'total_bk', 'shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths', 'zombies']


def load_source():
    with open(SOURCE, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def filter_and_write(rows, out_path, eta_filter=None, rho_filter=None, regime_filter=None):
    """Filter source rows by criteria, project to thesis schema, write CSV."""
    out_rows = []
    for r in rows:
        eta = float(r['eta'])
        rho = float(r['rho'])
        regime = r['fiscal_regime']
        if eta_filter is not None and abs(eta - eta_filter) > 1e-9: continue
        if rho_filter is not None and abs(rho - rho_filter) > 1e-9: continue
        if regime_filter is not None and regime not in regime_filter: continue
        out_rows.append({
            'seed': r['seed'],
            'fiscal_regime': regime,
            'rho': rho,
            'eta': eta,
            'omega': 0.58,
            'basis': 'equity',
            'inertia': 0.0,
            'total_bk': r['total_bk'],
            'shock': r['shock'],
            'rationing': r['rationing'],
            'repay': r['repay'],
            'contagion': r['contagion'],
            'fiscal_deaths': r['fiscal_deaths'],
            'zombies': r['zombies'],
        })
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=THESIS_COLS)
        w.writeheader()
        for r in out_rows: w.writerow(r)
    return len(out_rows)


def main():
    rows = load_source()
    print(f'Source: {SOURCE} ({len(rows)} rows)')

    # 1. thesis_rho_w58.csv: rho-sweep at eta=0, no-tax only
    out1 = os.path.join(SIM_DIR, 'thesis_rho_w58.csv')
    n = filter_and_write(rows, out1, eta_filter=0.0, regime_filter={'none'})
    print(f'  wrote {n} rows to thesis_rho_w58.csv')

    # 2. thesis_rho01_w58.csv: rho=0.1 sweep across (eta, regime)
    out2 = os.path.join(SIM_DIR, 'thesis_rho01_w58.csv')
    n = filter_and_write(rows, out2, rho_filter=0.10, regime_filter={'none', 'resolution_fund'})
    print(f'  wrote {n} rows to thesis_rho01_w58.csv')

    # 3-4. tau variants of thesis_rho (identical to canonical — tau-invariant at no-tax)
    for tau_tag in ['t5', 't6']:
        dst = os.path.join(SIM_DIR, f'thesis_rho_w58_{tau_tag}.csv')
        shutil.copy2(out1, dst)
        print(f'  copied thesis_rho_w58.csv -> thesis_rho_w58_{tau_tag}.csv')

    # 5-6. tau variants of thesis_rho01 (PLACEHOLDER copies — flagged for future)
    for tau_tag in ['t5', 't6']:
        dst = os.path.join(SIM_DIR, f'thesis_rho01_w58_{tau_tag}.csv')
        shutil.copy2(out2, dst)
        print(f'  copied thesis_rho01_w58.csv -> thesis_rho01_w58_{tau_tag}.csv (PLACEHOLDER)')

    print('\nDone. 6 thesis_rho files generated/copied.')


if __name__ == '__main__':
    main()
