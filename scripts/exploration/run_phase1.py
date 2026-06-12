"""Phase 1 grid: 6 fitness specs × 8 η × 3 seeds = 144 simulations.

Structural template: test_phase1_specs.py (SPECS list, run() pattern, metric
collection). File-handling: run_omega_sweep.py (safe_val NaN scrubbing, CSV
append + flush after each row, encoding='utf-8', parent ../Simulations/ dir).

Outputs:
    ../Simulations/phase1_grid.csv               — 144 rows + header
    ../Simulations/phase1_hub_series/*.json      — 18 files (η=0.1 only)

Run from the econo-ML-TBTF-clean directory:
    python run_phase1.py
"""
import os
import sys
import math
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc


# ── output paths (mirror run_omega_sweep.py) ─────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'Simulations')
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, 'phase1_grid.csv')
SERIES_DIR = os.path.join(SAVE_DIR, 'phase1_hub_series')
os.makedirs(SERIES_DIR, exist_ok=True)


# ── config (match test_phase1_specs.py CONFIG_BASE exactly) ──────────────────
# equity_heterogeneity / equity_cv intentionally omitted: the locked baseline
# and the byte-identity reference at seed 26462 (lender avg=1.19, max=6) both
# run with equity_heterogeneity=False (the Config default). run_omega_sweep.py
# uses heterogeneity=True for a different research question and is NOT the
# baseline for this grid.
CONFIG_BASE = dict(
    N=50, T=1000,
    omega=0.55, mu=0.7,
    rho=0.4,
    gamma_capital=0.10, alpha_collateral=0.05,
    beta=5, alfa=0.1,
    fiscal_regime="socialized_tax",
)

SPECS = [
    ("equity",    0.0),
    ("loan_book", 0.0),
    ("bilateral", 0.0),
    ("equity",    0.5),
    ("loan_book", 0.5),
    ("bilateral", 0.5),
]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
SEEDS = [26462, 26463, 26464]

HUB_SERIES_ETA = 0.1   # full time-series JSON dumped only at this eta

CSV_COLUMNS = [
    'seed', 'basis', 'inertia', 'eta',
    'total_bk', 'contagion', 'shock', 'rationing', 'repay', 'fiscal', 'zombies',
    'lender_avg_tenure', 'lender_max_tenure',
    'borrower_a_avg_tenure', 'borrower_a_max_tenure',
    'n_transitions', 'mortality_frac', 'clamp_fraction',
]


# ── helpers ──────────────────────────────────────────────────────────────────

def safe_val(v):
    """Convert a single value to CSV-safe string (NaN/Inf → empty), repr otherwise.
    Matches the run_omega_sweep.py pattern."""
    if v is None:
        return ''
    try:
        f = float(v)
        return '' if (math.isnan(f) or math.isinf(f)) else repr(f)
    except (TypeError, ValueError):
        return str(v)


def compute_tenure(best_lender, best_lender_gen, T):
    """RLE on the composite (id, generation) key. Returns (avg, max, n_runs).
    Matches the compute_hub_tenure_stats logic in gui_zombie.py without the
    import dependency, so this script stays standalone."""
    bl = list(best_lender[:T])
    bg = list(best_lender_gen[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    if not keys:
        return 0.0, 0, 0
    runs = []
    cur = keys[0]
    start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start)
            cur = keys[i]
            start = i
    runs.append(len(keys) - start)
    return sum(runs) / len(runs), max(runs), len(runs)


def decompose_hub_changes(best_lender, prev_alive, T):
    """Returns (n_transitions, mortality_fraction). Mirrors test_borrower_hub.py."""
    n_t, n_m = 0, 0
    for t in range(1, T):
        if best_lender[t] < 0 or best_lender[t - 1] < 0:
            continue
        if best_lender[t] == best_lender[t - 1]:
            continue
        n_t += 1
        if prev_alive[t] == 0:
            n_m += 1
    frac = (n_m / n_t) if n_t > 0 else 0.0
    return n_t, frac


# ── single simulation runner ─────────────────────────────────────────────────

def run_one(basis, inertia, eta, seed):
    """Run one simulation. Returns dict of metrics plus (_T, _stats) for the
    optional hub-series JSON dump.

    Wraps do_interest_rate with a closure that counts eq.6 evaluations and
    clamp hits. Counters live as INSTANCE attributes on the model (m._clamp_*)
    so each simulation has its own state — no module globals, no cross-run
    contamination. Lifted from test_clamp_fraction.py and adapted."""
    cfg = dict(CONFIG_BASE)
    cfg['eta_bailout'] = eta
    cfg['fitness_basis'] = basis
    cfg['fitness_inertia'] = inertia

    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)

    # Per-simulation clamp counters
    m._clamp_total = 0
    m._clamp_hits = 0

    orig_do_ir = m.do_interest_rate

    def wrapped_do_ir():
        orig_do_ir()
        # After do_interest_rate completes, recompute the unclamped eq.6
        # numerator/denominator for every (i, j) and compare to lender.C.
        # Same arithmetic as do_interest_rate so we measure identical evals.
        max_A = m.max_A_lagged if m.max_A_lagged > 0 else 1.0
        gamma = m.config.gamma_capital
        alpha = m.config.alpha_collateral
        eta_v = m.config.eta_bailout
        for bi in m.banks:
            if bi.failed:
                continue
            for bj in m.banks:
                if bi is bj or bj.failed:
                    continue
                p_j = 1 - bj.prob_surviving
                b_j = bj.A_lagged / max_A if m.max_A_lagged > 0 else 0
                # Skip the boundary cases (p_j<=0, p_j>=1, denom degenerate)
                # — for those the eq.6 cap doesn't apply.
                if p_j <= 0 or p_j >= 1.0 or (1 - b_j * eta_v) <= 0:
                    continue
                num = gamma * bi.E + p_j * (1 - b_j) * alpha * bj.A_lagged
                den = p_j * (1 - b_j * eta_v)
                unclamped = num / den
                m._clamp_total += 1
                if unclamped > bi.C + 1e-12:
                    m._clamp_hits += 1

    m.do_interest_rate = wrapped_do_ir

    m.simulate_full()
    m.finish()
    T = m.t
    stats = m.statistics

    # Bankruptcy-channel decomposition (full simulation totals)
    total_bk  = float(np.nansum(stats.bankruptcy[:T]))
    contagion = float(np.nansum(stats.bankruptcies_contagion[:T]))
    shock     = float(np.nansum(stats.bankruptcies_shock[:T]))
    rationing = float(np.nansum(stats.bankruptcies_rationing[:T]))
    repay     = float(np.nansum(stats.bankruptcies_repay[:T]))
    fiscal    = float(np.nansum(stats.bankruptcies_fiscal[:T]))
    zombies   = float(np.nansum(stats.fire_sale_survivors[:T]))

    lender_avg, lender_max, _ = compute_tenure(
        stats.best_lender, stats.best_lender_generation, T)
    borrower_avg, borrower_max, _ = compute_tenure(
        stats.top_A_bank, stats.top_A_generation, T)
    n_t, mort_frac = decompose_hub_changes(
        stats.best_lender, stats.previous_hub_alive, T)

    clamp_frac = (m._clamp_hits / m._clamp_total) if m._clamp_total > 0 else 0.0

    return {
        'total_bk': total_bk, 'contagion': contagion,
        'shock': shock, 'rationing': rationing, 'repay': repay,
        'fiscal': fiscal, 'zombies': zombies,
        'lender_avg_tenure': lender_avg, 'lender_max_tenure': lender_max,
        'borrower_a_avg_tenure': borrower_avg, 'borrower_a_max_tenure': borrower_max,
        'n_transitions': n_t, 'mortality_frac': mort_frac,
        'clamp_fraction': clamp_frac,
        '_T': T, '_stats': stats,
    }


# ── hub-series JSON dump (only at η = HUB_SERIES_ETA) ────────────────────────

def dump_hub_series(basis, inertia, seed, T, stats):
    """Dump full best_lender, best_lender_generation, top_A_bank, top_A_generation
    series to a single JSON. Composite-key generations included so downstream
    visualization can use the unambiguous (id, gen) tenure key."""
    inertia_str = ('%.1f' % inertia).replace('.', 'p')
    fname = f"hub_series_{basis}_{inertia_str}_seed{seed}.json"
    path = os.path.join(SERIES_DIR, fname)

    def to_int_list(arr):
        out = []
        for v in arr[:T]:
            try:
                iv = int(v)
                out.append(iv if iv >= 0 else None)
            except (TypeError, ValueError):
                out.append(None)
        return out

    payload = {
        'basis': basis,
        'inertia': inertia,
        'seed': seed,
        'T': int(T),
        'best_lender':            to_int_list(stats.best_lender),
        'best_lender_generation': to_int_list(stats.best_lender_generation),
        'top_A_bank':             to_int_list(stats.top_A_bank),
        'top_A_generation':       to_int_list(stats.top_A_generation),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)


# ── main grid ────────────────────────────────────────────────────────────────

def main():
    total = len(SPECS) * len(ETAS) * len(SEEDS)
    print(f"Phase 1 grid: {len(SPECS)} specs × {len(ETAS)} η × {len(SEEDS)} seeds = {total} sims")
    print(f"  ω={CONFIG_BASE['omega']}, ρ={CONFIG_BASE['rho']}, fiscal={CONFIG_BASE['fiscal_regime']}, T={CONFIG_BASE['T']}")
    print(f"CSV → {CSV_PATH}")
    print(f"Hub-series JSONs (η={HUB_SERIES_ETA}) → {SERIES_DIR}/")
    print()

    # Truncate any prior file and write the header
    with open(CSV_PATH, 'w', encoding='utf-8') as f:
        f.write(','.join(CSV_COLUMNS) + '\n')

    done = 0
    with open(CSV_PATH, 'a', encoding='utf-8') as csvf:
        for spec_idx, (basis, inertia) in enumerate(SPECS, 1):
            for eta_idx, eta in enumerate(ETAS, 1):
                for seed_idx, seed in enumerate(SEEDS, 1):
                    done += 1
                    print(f"running spec {spec_idx}/{len(SPECS)} ({basis} λ={inertia}), "
                          f"η {eta_idx}/{len(ETAS)} ({eta}), "
                          f"seed {seed_idx}/{len(SEEDS)} ({seed})  [{done}/{total}]",
                          flush=True)

                    r = run_one(basis, inertia, eta, seed)

                    row = [
                        seed, basis, inertia, eta,
                        r['total_bk'], r['contagion'],
                        r['shock'], r['rationing'], r['repay'], r['fiscal'], r['zombies'],
                        r['lender_avg_tenure'], r['lender_max_tenure'],
                        r['borrower_a_avg_tenure'], r['borrower_a_max_tenure'],
                        r['n_transitions'], r['mortality_frac'], r['clamp_fraction'],
                    ]
                    csvf.write(','.join(safe_val(v) for v in row) + '\n')
                    csvf.flush()

                    # Time-series dump only at the designated eta
                    if abs(eta - HUB_SERIES_ETA) < 1e-9:
                        dump_hub_series(basis, inertia, seed, r['_T'], r['_stats'])

    print(f"\nAll done. {total} rows → {os.path.basename(CSV_PATH)}")
    print(f"Hub-series dumps: {len(SPECS) * len(SEEDS)} JSONs in {os.path.basename(SERIES_DIR)}/")


if __name__ == '__main__':
    main()
