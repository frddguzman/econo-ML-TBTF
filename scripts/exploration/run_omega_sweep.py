"""
Omega sweep: run N=50, T=1000 at omega=0.50..0.60 (step 0.01)
with the parameters shown in the GUI screenshot, save per-period CSVs
in the same column format as downloadSignalCSV().

Run from the econo-ML-TBTF-clean directory:
    python run_omega_sweep.py
"""

import os
import sys
import math
import numpy as np

# ── make sure interbank.py is importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

# ── output directory ──────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'Simulations')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── columns to export (same order as SIGNAL_LABELS in index_zombie.html) ─────
STAT_NAMES = [
    'bankruptcy', 'bankruptcies_shock', 'bankruptcies_rationing',
    'bankruptcies_repay', 'bankruptcies_contagion', 'bankruptcies_fiscal',
    'fire_sale_survivors',
    'equity', 'interest_rate', 'liquidity', 'loans', 'B', 'rationing',
    'leverage', 'profits',
    'bailout_bill', 'bailout_count', 'tax_induced_failures',
    'resolution_fund_balance', 'fund_depleted_events', 'total_levy_collected',
    'fitness', 'best_lender_clients', 'num_banks',
    'active_borrowers', 'active_lenders', 'equity_lenders',
    'prob_bankruptcy', 'deposits', 'reserves', 'num_loans',
    'systemic_leverage', 'cascade_failures', 'bailout_tax_total',
    'bankruptcy_rationed',
    'best_lender', 'best_lender_fitness', 'best_lender_equity',
]

# ── fixed parameters (from GUI screenshot) ───────────────────────────────────
BASE_CONFIG = dict(
    N                  = 50,
    T                  = 1000,
    fiscal_regime      = 'none',
    eta_bailout        = 0.0,
    gamma_capital      = 0.10,
    alpha_collateral   = 0.05,
    rho                = 0.4,
    beta               = 5.0,
    alfa               = 0.1,
    mu                 = 0.7,
    equity_heterogeneity = True,
    equity_cv          = 0.5,
)
SEED      = 26462
ALGORITHM = 'Boltzmann'
LC_P      = 0.5   # p Attach
LC_M      = 1     # m Degree

# ── omega range ───────────────────────────────────────────────────────────────
omegas = [round(0.50 + i * 0.01, 2) for i in range(11)]   # 0.50 … 0.60


def safe_val(v):
    """Convert a single value to CSV-safe string (NaN/Inf → empty)."""
    if v is None:
        return ''
    try:
        f = float(v)
        return '' if (math.isnan(f) or math.isinf(f)) else repr(f)
    except (TypeError, ValueError):
        return str(v)


def extract_array(stats, name, T):
    """Pull a length-T array from statistics; fall back to zeros."""
    arr = getattr(stats, name, None)
    if arr is None:
        return [None] * T
    try:
        vals = arr[:T].tolist()
        return [None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                else v for v in vals]
    except Exception:
        return [None] * T


def run_and_save(omega):
    tag = f"omega={omega:.2f}_no-tax_eta0_hetero"
    print(f"  Running {tag} ...", end='', flush=True)

    cfg = dict(BASE_CONFIG)
    cfg['omega'] = omega

    model = interbank.Model()
    model.test = True
    model.configure(**cfg)
    model.config.lender_change = lc.determine_algorithm(ALGORITHM, p=LC_P, m=LC_M)
    model.initialize(seed=SEED, generate_plots=False)
    model.simulate_full()
    model.finish()

    T = model.config.T
    stats = model.statistics

    # build column data
    columns = {'time': list(range(T))}
    for name in STAT_NAMES:
        columns[name] = extract_array(stats, name, T)

    # write CSV
    header = ['time'] + STAT_NAMES
    rows = [','.join(header)]
    for t in range(T):
        row = [str(t)] + [safe_val(columns[name][t]) for name in STAT_NAMES]
        rows.append(','.join(row))

    csv_path = os.path.join(SAVE_DIR, tag + '.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))

    # quick hub summary
    hub_ids = columns['best_lender']
    turnovers = sum(1 for i in range(1, T) if hub_ids[i] != hub_ids[i-1]
                    and hub_ids[i] is not None and hub_ids[i-1] is not None)
    runs = []
    cur = 1
    for i in range(1, T):
        if hub_ids[i] == hub_ids[i-1]: cur += 1
        else: runs.append(cur); cur = 1
    runs.append(cur)
    avg_t = round(sum(runs) / len(runs), 1)
    max_t = max(runs)
    cli   = columns['best_lender_clients']
    avg_c = round(sum(v for v in cli if v is not None) / T, 1)
    bk    = int(sum(v for v in columns['bankruptcy'] if v is not None))

    print(f"  done | turnovers={turnovers}  avg_tenure={avg_t}  max_tenure={max_t}"
          f"  hub_clients_avg={avg_c}  bankruptcies={bk}")
    print(f"         saved: {os.path.basename(csv_path)}")


if __name__ == '__main__':
    print(f"Omega sweep: {omegas}")
    print(f"Seed={SEED}, N=50, T=1000, beta=5, mu=0.7, fiscal=none, eta=0, hetero CV=0.5\n")
    for omega in omegas:
        run_and_save(omega)
    print("\nAll done.")
