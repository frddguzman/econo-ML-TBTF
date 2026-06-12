"""Phase 0: synthesize thesis_lehman_w58_t6.csv and thesis_lehman_w58_t5.csv
from existing sweep data — no new sims required.

Logic:
  - For fiscal_regime in {'none', 'socialized_tax'}: τ_fund is irrelevant
    (the fund mechanism doesn't engage), so the τ=1e-4 rows from
    thesis_lehman_w58.csv are valid for any τ.
  - For fiscal_regime == 'resolution_fund': substitute the τ-specific rows
    from sweep_w58_tau_low_raw.csv (filtered by tau column).

Schemas:
  thesis_lehman_w58.csv:
    seed, fiscal_regime, rho, eta, omega, mu, gamma, basis, inertia,
    total_bk, shock, rationing, repay, contagion, fiscal_deaths, zombies
  sweep_w58_tau_low_raw.csv:
    tau, eta, seed, total_bk, shock, rationing, repay, contagion,
    fiscal_deaths, zombies, avg_cli, avg_ten, max_ten, turnovers
    → expand to thesis schema with rho=0.4, omega=0.58, mu=0.7, gamma=0.10,
    basis='equity', inertia=0.0, fiscal_regime='resolution_fund'.
"""
import os
import csv
from pathlib import Path

# Output / simulation data directory. Override via TBTF_SIM_DIR env var.
# Default: a sibling 'Simulations/' folder next to the repo root.
SIM = str(Path(os.environ.get(
    'TBTF_SIM_DIR',
    str(Path(__file__).resolve().parent.parent / 'Simulations'),
)))

# Project root (the repo directory itself).
PROJ = str(Path(__file__).resolve().parent)

THESIS_W58 = os.path.join(SIM, 'thesis_lehman_w58.csv')
TAU_LOW    = os.path.join(PROJ, 'sweep_w58_tau_low_raw.csv')

OUT_HEADER = ['seed', 'fiscal_regime', 'rho', 'eta', 'omega', 'mu', 'gamma',
              'basis', 'inertia', 'total_bk', 'shock', 'rationing', 'repay',
              'contagion', 'fiscal_deaths', 'zombies']

# Constants for the w58 cell (read from thesis_lehman_w58.csv first row)
CONST = dict(rho=0.4, omega=0.58, mu=0.7, gamma=0.1, basis='equity', inertia=0.0)


def synthesize(tau, out_path):
    # Load the no-tax + socialized rows from the τ=1e-4 thesis sweep
    nontax_rows = []
    with open(THESIS_W58, 'r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r['fiscal_regime'] in ('none', 'socialized_tax'):
                nontax_rows.append(r)

    # Load the resolution_fund rows for this τ from the tau-low sweep
    fund_rows = []
    tau_str = repr(float(tau))   # match how the CSV stored the float
    with open(TAU_LOW, 'r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            if abs(float(r['tau']) - float(tau)) < 1e-12:
                fund_rows.append({
                    'seed':           r['seed'],
                    'fiscal_regime':  'resolution_fund',
                    'rho':            CONST['rho'],
                    'eta':            r['eta'],
                    'omega':          CONST['omega'],
                    'mu':             CONST['mu'],
                    'gamma':          CONST['gamma'],
                    'basis':          CONST['basis'],
                    'inertia':        CONST['inertia'],
                    'total_bk':       r['total_bk'],
                    'shock':          r['shock'],
                    'rationing':      r['rationing'],
                    'repay':          r['repay'],
                    'contagion':      r['contagion'],
                    'fiscal_deaths':  r['fiscal_deaths'],
                    'zombies':        r['zombies'],
                })

    # Write combined CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=OUT_HEADER)
        w.writeheader()
        for r in nontax_rows:
            w.writerow({k: r.get(k, '') for k in OUT_HEADER})
        for r in fund_rows:
            w.writerow({k: r.get(k, '') for k in OUT_HEADER})

    n_nt = sum(1 for r in nontax_rows if r['fiscal_regime'] == 'none')
    n_st = sum(1 for r in nontax_rows if r['fiscal_regime'] == 'socialized_tax')
    n_rf = len(fund_rows)
    print(f'  tau={tau:.0e}: wrote {out_path} ({n_nt} nt + {n_st} st + {n_rf} rf rows)')


def main():
    print('Synthesizing w58 low-tau thesis sweeps')
    synthesize(1e-6, os.path.join(SIM, 'thesis_lehman_w58_t6.csv'))
    synthesize(1e-5, os.path.join(SIM, 'thesis_lehman_w58_t5.csv'))
    print('Done.')


if __name__ == '__main__':
    main()
