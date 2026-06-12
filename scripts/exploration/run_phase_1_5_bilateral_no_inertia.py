"""Phase 1.5 ω-sweep: bilateral fitness, NO inertia (λ=0.0).

Companion to run_phase_1_5.py (which uses bilateral λ=0.5). Same grid, same
parallelism, same metric collection — only the CONFIG_BASE fitness_inertia
and the output CSV filename differ.

Grid: 6 omegas × 5 etas × 3 seeds = 90 sims.

Run from the econo-ML-TBTF-clean directory:
    python run_phase_1_5_bilateral_no_inertia.py
"""
import os
import sys
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc


# ── output paths ─────────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'Simulations')
os.makedirs(SAVE_DIR, exist_ok=True)
CSV_PATH = os.path.join(SAVE_DIR, 'phase1_5_omega_sweep_bilateral_l0.csv')


# ── config ───────────────────────────────────────────────────────────────────
# Bilateral fitness, λ=0 (no inertia). omega and eta swept; seed varied.
CONFIG_BASE = dict(
    N=50, T=1000,
    mu=0.7,
    rho=0.4,
    gamma_capital=0.10, alpha_collateral=0.05,
    beta=5, alfa=0.1,
    fiscal_regime="socialized_tax",
    fitness_basis="bilateral",
    fitness_inertia=0.0,
)

OMEGAS = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
ETAS = [0.0, 0.1, 0.3, 0.5, 0.9]
SEEDS = [26462, 26463, 26464]

CSV_COLUMNS = [
    'seed', 'basis', 'inertia', 'omega', 'eta',
    'total_bk', 'contagion', 'shock', 'rationing', 'repay', 'fiscal', 'zombies',
    'lender_avg_tenure', 'lender_max_tenure',
    'borrower_a_avg_tenure', 'borrower_a_max_tenure',
    'n_transitions', 'mortality_frac', 'clamp_fraction',
]


# ── helpers ──────────────────────────────────────────────────────────────────

def safe_val(v):
    if v is None:
        return ''
    try:
        f = float(v)
        return '' if (math.isnan(f) or math.isinf(f)) else repr(f)
    except (TypeError, ValueError):
        return str(v)


def compute_tenure(best_lender, best_lender_gen, T):
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


# ── worker ───────────────────────────────────────────────────────────────────

def run_one(omega, eta, seed):
    cfg = dict(CONFIG_BASE)
    cfg['omega'] = omega
    cfg['eta_bailout'] = eta

    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)

    m._clamp_total = 0
    m._clamp_hits = 0

    orig_do_ir = m.do_interest_rate

    def wrapped_do_ir():
        orig_do_ir()
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
        'seed': seed, 'omega': omega, 'eta': eta,
        'basis': cfg['fitness_basis'], 'inertia': cfg['fitness_inertia'],
        'total_bk': total_bk, 'contagion': contagion,
        'shock': shock, 'rationing': rationing, 'repay': repay,
        'fiscal': fiscal, 'zombies': zombies,
        'lender_avg_tenure': lender_avg, 'lender_max_tenure': lender_max,
        'borrower_a_avg_tenure': borrower_avg, 'borrower_a_max_tenure': borrower_max,
        'n_transitions': n_t, 'mortality_frac': mort_frac,
        'clamp_fraction': clamp_frac,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    tasks = [(omega, eta, seed)
             for omega in OMEGAS
             for eta in ETAS
             for seed in SEEDS]
    total = len(tasks)
    workers = max(1, (os.cpu_count() or 2) - 1)

    print(f"Phase 1.5 omega sweep (bilateral λ=0.0):")
    print(f"  {len(OMEGAS)} omegas × {len(ETAS)} etas × {len(SEEDS)} seeds = {total} sims")
    print(f"  workers: {workers} (cpu_count - 1)")
    print(f"  CSV → {CSV_PATH}")
    print()

    with open(CSV_PATH, 'w', encoding='utf-8') as f:
        f.write(','.join(CSV_COLUMNS) + '\n')

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, omega, eta, seed): (omega, eta, seed)
            for (omega, eta, seed) in tasks
        }

        with open(CSV_PATH, 'a', encoding='utf-8') as csvf:
            for future in as_completed(futures):
                omega, eta, seed = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    done += 1
                    print(f"[{done}/{total}] ω={omega} η={eta} seed={seed}  FAILED: {e}",
                          flush=True)
                    continue

                done += 1
                print(f"[{done}/{total}] ω={omega:.2f} η={eta} seed={seed}  "
                      f"lender_avg={r['lender_avg_tenure']:.2f} max={r['lender_max_tenure']}  "
                      f"total_bk={int(r['total_bk'])}  contagion={int(r['contagion'])}",
                      flush=True)

                row = [r[col] for col in CSV_COLUMNS]
                csvf.write(','.join(safe_val(v) for v in row) + '\n')
                csvf.flush()

    print(f"\nAll done. {total} rows → {os.path.basename(CSV_PATH)}")


if __name__ == '__main__':
    main()
