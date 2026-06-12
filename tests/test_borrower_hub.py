"""One-shot test: is argmax(A_lagged) identity persistent?

Runs T=1000 at the locked baseline (omega=0.55, eta=0.1, rho=0.4,
fiscal_regime=socialized_tax) for three seeds and reports true tenure
on three identity series, using the composite (id, generation) key.
"""
import sys

import interbank
import interbank_lenderchange as lc
from gui_zombie import compute_hub_tenure_stats


CONFIG = dict(
    N=50, T=1000,
    omega=0.50, mu=0.7,
    eta_bailout=0.1, rho=0.4,
    gamma_capital=0.10, alpha_collateral=0.05,
    beta=5, alfa=0.1,
    fiscal_regime="socialized_tax",
)
SEEDS = [26462, 26463, 26464]


def decompose_hub_changes(best_lender, prev_alive, T):
    """For each transition where best_lender[t] != best_lender[t-1] and both
    are valid, classify as mortality-driven (prev hub bank generation ticked)
    or fitness-driven (still alive, just lost the position).
    Returns (n_transitions, n_mortality, n_fitness, mortality_frac)."""
    n_t, n_m, n_f = 0, 0, 0
    for t in range(1, T):
        if best_lender[t] < 0 or best_lender[t - 1] < 0:
            continue
        if best_lender[t] == best_lender[t - 1]:
            continue
        n_t += 1
        flag = prev_alive[t]
        if flag == 0:
            n_m += 1
        elif flag == 1:
            n_f += 1
    frac = (n_m / n_t) if n_t > 0 else 0.0
    return n_t, n_m, n_f, frac


def run_one(seed):
    m = interbank.Model()
    m.test = True
    m.configure(**CONFIG)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    t = m.t

    def stats(id_arr, gen_arr):
        return compute_hub_tenure_stats(
            getattr(m.statistics, id_arr, None),
            getattr(m.statistics, gen_arr, None),
            t,
        )

    borrower_a = stats("top_A_bank", "top_A_generation")
    top_loan   = stats("top_borrower_bank", "top_borrower_generation")
    lender     = stats("best_lender", "best_lender_generation")
    decomp = decompose_hub_changes(
        m.statistics.best_lender, m.statistics.previous_hub_alive, t,
    )
    return borrower_a, top_loan, lender, decomp


def fmt(stats):
    return "avg=%5.2f max=%4d" % (stats["avg_tenure_true"], stats["max_tenure_true"])


def main():
    sums = {"borrower_a": 0.0, "top_loan": 0.0, "lender": 0.0,
            "mort_frac": 0.0, "transitions": 0}
    for seed in SEEDS:
        ba, tl, ln, dc = run_one(seed)
        n_t, n_m, n_f, frac = dc
        print("seed=%d  borrower-A: %s  |  top-loan: %s  |  lender: %s"
              % (seed, fmt(ba), fmt(tl), fmt(ln)))
        print("           hub changes: %d total | %d mortality (%.1f%%) | %d fitness (%.1f%%)"
              % (n_t, n_m, frac * 100, n_f, (1 - frac) * 100 if n_t else 0))
        sums["borrower_a"] += ba["avg_tenure_true"]
        sums["top_loan"]   += tl["avg_tenure_true"]
        sums["lender"]     += ln["avg_tenure_true"]
        sums["mort_frac"] += frac
        sums["transitions"] += n_t
    n = len(SEEDS)
    print("mean         borrower-A: avg=%5.2f         |  top-loan: avg=%5.2f         |  lender: avg=%5.2f"
          % (sums["borrower_a"] / n, sums["top_loan"] / n, sums["lender"] / n))
    print("mean         mortality-driven: %.1f%% of all hub changes (across %d total transitions)"
          % (100 * sums["mort_frac"] / n, sums["transitions"]))


if __name__ == "__main__":
    main()
