"""Phase 0 diagnostic: does cranking up Boltzmann's gamma momentum produce
hub formation by itself? If yes, Phase 1 toggle implementation is unnecessary.

Three sims at omega=0.5, eta=0.1, rho=0.4, socialized_tax, T=1000, seed=26462.
gamma in {0.0, 0.5, 0.9}.

Reports per gamma:
- raw lender hub avg/max true tenure (composite key)
- borrower-A avg/max (informational)
- total bankruptcies, contagion column
- mortality-driven % of hub changes
"""
import numpy as np

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
SEED = 26462
GAMMAS = [0.0, 0.5, 0.9]


def decompose(best_lender, prev_alive, T):
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


def run_one(gamma):
    m = interbank.Model()
    m.test = True
    m.configure(**CONFIG)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.config.lender_change.gamma = gamma  # override momentum
    m.initialize(seed=SEED, generate_plots=False)
    m.simulate_full()
    m.finish()
    t = m.t

    def stats(id_arr, gen_arr):
        return compute_hub_tenure_stats(
            getattr(m.statistics, id_arr, None),
            getattr(m.statistics, gen_arr, None),
            t,
        )

    lender = stats("best_lender", "best_lender_generation")
    borrower_a = stats("top_A_bank", "top_A_generation")
    n_t, n_m, n_f, mort_frac = decompose(
        m.statistics.best_lender, m.statistics.previous_hub_alive, t,
    )
    total_bk = int(np.nansum(m.statistics.bankruptcy[:t]))
    contagion = int(np.nansum(m.statistics.bankruptcies_contagion[:t]))
    return {
        'lender_avg': lender['avg_tenure_true'],
        'lender_max': lender['max_tenure_true'],
        'borrower_avg': borrower_a['avg_tenure_true'],
        'borrower_max': borrower_a['max_tenure_true'],
        'transitions': n_t,
        'mortality_frac': mort_frac,
        'total_bk': total_bk,
        'contagion': contagion,
    }


def main():
    print("Phase 0: gamma momentum diagnostic at omega=0.5, eta=0.1, T=1000, seed=%d" % SEED)
    print()
    print("%-8s %-12s %-12s %-12s %-13s %-12s" % (
        "gamma", "lender", "borrower-A", "transitions", "total_bk", "contagion",
    ))
    print("-" * 75)
    rows = []
    for g in GAMMAS:
        r = run_one(g)
        rows.append((g, r))
        print("%-8.2f avg=%4.2f max=%2d  avg=%4.2f max=%2d  %4d (%.0f%% mort)  %5d        %5d" % (
            g,
            r['lender_avg'], r['lender_max'],
            r['borrower_avg'], r['borrower_max'],
            r['transitions'], 100 * r['mortality_frac'],
            r['total_bk'],
            r['contagion'],
        ))
    print()
    print("Decision (per plan):")
    print("  PASS if any gamma: lender avg >= 5 AND lender max >= 15 AND contagion > 1000")
    for g, r in rows:
        passes = (r['lender_avg'] >= 5
                  and r['lender_max'] >= 15
                  and r['contagion'] > 1000)
        print("  gamma=%.2f: %s" % (g, "PASS" if passes else "FAIL"))


if __name__ == "__main__":
    main()
