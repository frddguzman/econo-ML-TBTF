"""Two-phase runner for tomorrow's dashboard.

Phase 1 (`python dump_dashboard_runs.py search`)
    Constrained maximin seed search across 30 candidates × 5 etas × 2 candidate
    cells (Cand A = equity ω=0.53; Cand B = bilateral ω=0.52). Reports the
    chosen seed and HALTS — the 6-cell dump is a separate run after the user
    confirms the chosen seed.

Phase 2 (`python dump_dashboard_runs.py dump <seed>`)
    Runs all 6 dashboard configurations with the given seed and writes one
    per-period CSV per tag. Column set is STAT_NAMES_EXT (run_omega_sweep.py's
    STAT_NAMES plus Phase 1 borrower-A and decomposition columns) so Hub
    Tracker can drive borrower-A views and the mortality decomposition column
    is non-empty.

Run from the econo-ML-TBTF-clean directory.
"""
import os
import sys
import math
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc


# ── output directory (mirrors run_omega_sweep.py) ────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'Simulations')
os.makedirs(SAVE_DIR, exist_ok=True)


# ── column set: run_omega_sweep.STAT_NAMES + Phase 1 additions (Patch 2) ────
STAT_NAMES_BASE = [
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
STAT_NAMES_EXT = STAT_NAMES_BASE + [
    'top_A_bank', 'top_A_generation', 'top_A_value',
    'best_lender_generation',
    'previous_hub_alive',
]


# ── candidate cells and search pool ──────────────────────────────────────────
HEADLINE_CELLS = [
    ('equity',    0.53),   # Cand A
    ('bilateral', 0.52),   # Cand B
]
ETAS_SUBGRID = [0.0, 0.1, 0.3, 0.5, 0.9]
SEED_POOL = list(range(26462, 26492))   # 30 seeds


# ── 6 dashboard configurations (OG plan table) ───────────────────────────────
DASH_CONFIGS = [
    ('baseline_eta01',       'equity',    0.50, 0.10),
    ('baseline_eta085',      'equity',    0.50, 0.85),
    ('equity_w53_eta01',     'equity',    0.53, 0.10),
    ('equity_w53_eta085',    'equity',    0.53, 0.85),
    ('bilateral_w52_eta01',  'bilateral', 0.52, 0.10),
    ('bilateral_w52_eta085', 'bilateral', 0.52, 0.85),
]

# ── 27-cell grid: 3 candidates × 3 fiscal regimes × 3 η ──────────────────────
# Tag schema: dash_<cand>_<regime>_<eta>.csv
# cand:   bsl=baseline (eq, ω=0.50)  a=Cand A (eq, ω=0.53)  b=Cand B (bil, ω=0.52)
# regime: nt=no_tax  st=socialized_tax  rf=resolution_fund
# eta:    e0=0.0  e01=0.1  e085=0.85
CAND_DEFS = [
    ('bsl', 'equity',    0.50),
    ('a',   'equity',    0.53),
    ('b',   'bilateral', 0.52),
]
REGIME_DEFS = [
    ('nt', 'none'),
    ('st', 'socialized_tax'),
    ('rf', 'resolution_fund'),
]
ETA_DEFS = [
    ('e0',   0.00),
    ('e01',  0.10),
    ('e085', 0.85),
]
DASH27_CONFIGS = []
for c_tag, basis, omega in CAND_DEFS:
    for r_tag, regime in REGIME_DEFS:
        for e_tag, eta in ETA_DEFS:
            tag = f"{c_tag}_{r_tag}_{e_tag}"
            DASH27_CONFIGS.append((tag, basis, omega, eta, regime))


# ── shared simulation config (matches phase1 baseline) ───────────────────────
# fund_levy_rate=0.0001 matches gui_zombie.py default. interbank.py's 0.005 is 50×
# too high — at that level the levy alone bankrupts every bank within ~20 periods
# and breaks the resolution_fund mechanism (Lehman zone disappears).
def make_config(basis, omega, eta, regime="socialized_tax"):
    return dict(
        N=50, T=1000,
        mu=0.7,
        omega=omega,
        eta_bailout=eta,
        rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=regime,
        fund_levy_rate=0.0001,
        fitness_basis=basis,
        fitness_inertia=0.0,
    )


def run_sim(basis, omega, eta, seed, regime="socialized_tax"):
    """Run one simulation and return the model (with .statistics populated)."""
    cfg = make_config(basis, omega, eta, regime=regime)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    return m


def compute_lender_max(stats, T):
    """RLE on composite (best_lender, best_lender_generation) key, return
    longest run length. Mirrors compute_tenure in run_phase1.py."""
    bl = list(stats.best_lender[:T])
    bg = list(stats.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    if not keys:
        return 0
    runs = []
    cur = keys[0]
    start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start)
            cur = keys[i]
            start = i
    runs.append(len(keys) - start)
    return max(runs)


# ── Phase 1: seed search ─────────────────────────────────────────────────────

def evaluate_seed(seed):
    """Run all 10 (cell × eta) combos for one seed.
    Returns (seed, dict[(basis, omega, eta) -> (total_bk, lender_max)])."""
    out = {}
    for basis, omega in HEADLINE_CELLS:
        for eta in ETAS_SUBGRID:
            m = run_sim(basis, omega, eta, seed)
            T = m.t
            total_bk = int(np.nansum(m.statistics.bankruptcy[:T]))
            l_max = compute_lender_max(m.statistics, T)
            out[(basis, omega, eta)] = (total_bk, l_max)
    return seed, out


def eligible(out):
    """Constraint: at BOTH candidate cells, total_bk(η=0.1) < total_bk(η=0)."""
    for basis, omega in HEADLINE_CELLS:
        bk0  = out[(basis, omega, 0.0)][0]
        bk01 = out[(basis, omega, 0.1)][0]
        if bk01 >= bk0:
            return False
    return True


def cell_metrics(out, basis, omega):
    """Return (max_tenure_eta01, eta_star_depth_pct) for a given cell."""
    bk0   = out[(basis, omega, 0.0)][0]
    bk01  = out[(basis, omega, 0.1)][0]
    l_max = out[(basis, omega, 0.1)][1]
    depth_pct = 100.0 * (bk01 - bk0) / bk0 if bk0 > 0 else 0.0
    return l_max, depth_pct


def phase_search():
    workers = max(1, min(6, (os.cpu_count() or 2) - 1))
    print(f"Seed search: {len(SEED_POOL)} seeds × {len(ETAS_SUBGRID)} etas × "
          f"{len(HEADLINE_CELLS)} cells = "
          f"{len(SEED_POOL) * len(ETAS_SUBGRID) * len(HEADLINE_CELLS)} sims")
    print(f"Workers: {workers}")
    print()

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for seed, out in pool.map(evaluate_seed, SEED_POOL):
            results[seed] = out
            print(f"  seed {seed} done")

    eligible_seeds = {s: o for s, o in results.items() if eligible(o)}
    print()
    print(f"Eligible seeds (η* sign-stable at both cells): "
          f"{len(eligible_seeds)} of {len(SEED_POOL)}")

    if not eligible_seeds:
        print()
        print("FALLBACK: zero seeds passed the η* constraint at both cells.")
        print("Stopping here. Discuss with user before relaxing the constraint.")
        sys.exit(1)

    # Score: maximin on max_tenure at η=0.1 across both cells.
    # Tiebreaker: deeper η* depth at Cell B (bilateral, the headline finding).
    def score(out):
        m_A, _   = cell_metrics(out, *HEADLINE_CELLS[0])
        m_B, dB  = cell_metrics(out, *HEADLINE_CELLS[1])
        # primary: maximin (higher is better)
        # tiebreaker: more negative dB = deeper η* (so use -dB; higher better)
        return (min(m_A, m_B), -dB)

    chosen = max(eligible_seeds, key=lambda s: score(eligible_seeds[s]))

    # Report
    print()
    print("=" * 70)
    print(f"CHOSEN SEED: {chosen}")
    print("=" * 70)
    print()
    print(f"At chosen seed {chosen}:")
    for basis, omega in HEADLINE_CELLS:
        l_max, depth = cell_metrics(eligible_seeds[chosen], basis, omega)
        print(f"  {basis:<10} ω={omega}  η=0.1:  "
              f"lender_max = {l_max:>3d}    η* depth = {depth:+6.2f}%")

    print()
    print("Comparison vs OG hand-pick seed 26463:")
    if 26463 in results:
        og = results[26463]
        og_eligible = "PASS" if eligible(og) else "FAIL"
        print(f"  (η* eligibility at seed 26463: {og_eligible})")
        for basis, omega in HEADLINE_CELLS:
            l_max, depth = cell_metrics(og, basis, omega)
            print(f"  {basis:<10} ω={omega}  η=0.1:  "
                  f"lender_max = {l_max:>3d}    η* depth = {depth:+6.2f}%")
    else:
        print("  (seed 26463 not in pool; skipping comparison)")

    print()
    print(f"Top-5 eligible seeds by score:")
    ranked = sorted(eligible_seeds.keys(),
                    key=lambda s: score(eligible_seeds[s]),
                    reverse=True)[:5]
    for s in ranked:
        m_A, dA = cell_metrics(eligible_seeds[s], *HEADLINE_CELLS[0])
        m_B, dB = cell_metrics(eligible_seeds[s], *HEADLINE_CELLS[1])
        print(f"  seed {s}: A max={m_A:>2d} dA={dA:+5.2f}%  |  "
              f"B max={m_B:>2d} dB={dB:+5.2f}%  |  min={min(m_A, m_B)}")

    print()
    print(f"HALT. Confirm chosen seed = {chosen}, then run:")
    print(f"  python dump_dashboard_runs.py dump {chosen}")


# ── Phase 2: 6 per-period dumps with the confirmed seed ──────────────────────

def safe_val(v):
    if v is None:
        return ''
    try:
        f = float(v)
        return '' if (math.isnan(f) or math.isinf(f)) else repr(f)
    except (TypeError, ValueError):
        return str(v)


def extract_array(stats, name, T):
    arr = getattr(stats, name, None)
    if arr is None:
        return [None] * T
    try:
        vals = arr[:T].tolist()
        return [None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                else v for v in vals]
    except Exception:
        return [None] * T


def dump_one(args):
    """Worker for the dump phase."""
    tag, basis, omega, eta, seed = args
    m = run_sim(basis, omega, eta, seed)
    T = m.t
    stats = m.statistics

    columns = {'time': list(range(T))}
    empty_cols = []
    for name in STAT_NAMES_EXT:
        col = extract_array(stats, name, T)
        columns[name] = col
        if all(v is None or v == 0 for v in col):
            empty_cols.append(name)

    header = ['time'] + STAT_NAMES_EXT
    rows = [','.join(header)]
    for t in range(T):
        rows.append(','.join([str(t)] + [safe_val(columns[name][t])
                                          for name in STAT_NAMES_EXT]))

    csv_path = os.path.join(SAVE_DIR, f'dash_{tag}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))

    return tag, T, compute_lender_max(stats, T), empty_cols


def phase_dump(seed):
    workers = max(1, min(len(DASH_CONFIGS), (os.cpu_count() or 2) - 1))
    print(f"Dumping {len(DASH_CONFIGS)} per-period CSVs with seed={seed} "
          f"({workers} workers)")
    print()

    jobs = [(tag, basis, omega, eta, seed)
            for tag, basis, omega, eta in DASH_CONFIGS]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(dump_one, jobs))

    print(f"{'tag':<25} {'T':>5} {'lender_max':>11}  empty_cols")
    print("-" * 70)
    all_empty = set()
    for tag, T, l_max, empty in results:
        empty_str = ','.join(empty) if empty else '-'
        print(f"{tag:<25} {T:>5} {l_max:>11}  {empty_str[:50]}")
        all_empty.update(empty)

    print()
    print(f"Wrote {len(results)} CSVs to {SAVE_DIR}")
    if all_empty:
        print(f"WARNING: columns empty in at least one dump: {sorted(all_empty)}")
        print("  (These statistics may not be tracked under the spec used.)")


# ── Phase 3: 27-cell dump (per-period + bank-detail) ─────────────────────────

BANK_DETAIL_COLS = ['t', 'bank_id', 'equity', 'fitness', 'p_j', 'b_j',
                    'interest_rate', 'loan', 'clients', 'is_hub']


def dump_one27(args):
    """Worker for the 27-cell dump. Writes both:
    - Simulations/dash_<tag>.csv         : per-period time series (STAT_NAMES_EXT)
    - Simulations/bank_detail_<tag>.csv  : per-(bank, period) rows
    """
    tag, basis, omega, eta, regime, seed = args
    m = run_sim(basis, omega, eta, seed, regime=regime)
    T = m.t
    stats = m.statistics

    # ── Per-period CSV ─────────────────────────────────────────────
    columns = {'time': list(range(T))}
    empty_cols = []
    for name in STAT_NAMES_EXT:
        col = extract_array(stats, name, T)
        columns[name] = col
        if all(v is None or v == 0 for v in col):
            empty_cols.append(name)

    header = ['time'] + STAT_NAMES_EXT
    rows = [','.join(header)]
    for t in range(T):
        rows.append(','.join([str(t)] + [safe_val(columns[name][t])
                                          for name in STAT_NAMES_EXT]))
    csv_path = os.path.join(SAVE_DIR, f'dash_{tag}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))

    # ── Bank-detail CSV ─────────────────────────────────────────────
    bd_rows = getattr(stats, '_bank_rows', []) or []
    bd_header = ','.join(BANK_DETAIL_COLS)
    bd_lines = [bd_header]
    for r in bd_rows:
        bd_lines.append(','.join(safe_val(r.get(c)) for c in BANK_DETAIL_COLS))
    bd_path = os.path.join(SAVE_DIR, f'bank_detail_{tag}.csv')
    with open(bd_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bd_lines))

    return tag, T, compute_lender_max(stats, T), len(bd_rows), empty_cols


def phase_dump27(seed):
    workers = max(1, min(len(DASH27_CONFIGS), (os.cpu_count() or 2) - 1))
    print(f"27-cell dump (seed={seed}): {len(DASH27_CONFIGS)} cells × 2 CSVs each "
          f"({workers} workers)")
    print()

    jobs = [(tag, basis, omega, eta, regime, seed)
            for tag, basis, omega, eta, regime in DASH27_CONFIGS]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(dump_one27, jobs))

    print(f"{'tag':<22} {'T':>5} {'lender_max':>11} {'bank_rows':>11}  empty_cols")
    print("-" * 75)
    all_empty = set()
    for tag, T, l_max, n_bd, empty in results:
        empty_str = ','.join(empty) if empty else '-'
        print(f"{tag:<22} {T:>5} {l_max:>11} {n_bd:>11}  {empty_str[:38]}")
        all_empty.update(empty)

    print()
    print(f"Wrote {len(results)} per-period CSVs + {len(results)} bank-detail CSVs to {SAVE_DIR}")
    if all_empty:
        print(f"NOTE: empty in at least one dump: {sorted(all_empty)}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('search')
    p_dump = sub.add_parser('dump')
    p_dump.add_argument('seed', type=int)
    p_dump27 = sub.add_parser('dump27')
    p_dump27.add_argument('seed', type=int)
    args = parser.parse_args()

    if args.cmd == 'search':
        phase_search()
    elif args.cmd == 'dump':
        phase_dump(args.seed)
    elif args.cmd == 'dump27':
        phase_dump27(args.seed)


if __name__ == '__main__':
    main()
