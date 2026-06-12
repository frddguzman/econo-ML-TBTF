"""Phase 1 R1 diagnostic: characterize how often eq.6's min(..., C_i) clamp
fires, and the distribution of b_j at the locked baseline. Informs how
much the bilateral spec's regime-flip effect (point 2 of R1) actually
contributes.

Monkey-patches do_interest_rate via a counter wrapper around eq.6.
Single seed, T=1000, omega=0.5, eta=0.1.
"""
import numpy as np

import interbank
import interbank_lenderchange as lc


CONFIG = dict(
    N=50, T=1000,
    omega=0.50, mu=0.7,
    eta_bailout=0.1, rho=0.4,
    gamma_capital=0.10, alpha_collateral=0.05,
    beta=5, alfa=0.1,
    fiscal_regime="socialized_tax",
)
SEED = 26462


# Counters captured via closure
total_evals = 0
clamp_hits = 0
nonclamp_hits = 0
b_j_samples = []


def main():
    global total_evals, clamp_hits, nonclamp_hits, b_j_samples

    m = interbank.Model()
    m.test = True
    m.configure(**CONFIG)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=SEED, generate_plots=False)

    # Wrap do_interest_rate to instrument eq.6 evals.
    # We count by sampling AFTER do_interest_rate completes each period:
    # for each (i, j) pair, compare bank_i.L_ij_max[j] to bank_i.C at the
    # moment of computation. Approximation: bank.C may have changed by the
    # next sample point, so we record at the end of do_interest_rate.
    orig_do_ir = m.do_interest_rate

    def wrapped_do_ir():
        global total_evals, clamp_hits, nonclamp_hits, b_j_samples
        orig_do_ir()
        # Snapshot all (i, j) L_ij_max values vs the eq.6 unclamped value
        # by recomputing the unclamped numerator/denominator. This is the
        # only way to know whether the clamp actually fired or not.
        max_A = m.max_A_lagged if m.max_A_lagged > 0 else 1.0
        max_E = m.maxE if m.maxE > 0 else 1.0
        gamma = m.config.gamma_capital
        alpha = m.config.alpha_collateral
        eta = m.config.eta_bailout
        for i, bi in enumerate(m.banks):
            if bi.failed:
                continue
            for j, bj in enumerate(m.banks):
                if i == j or bj.failed:
                    continue
                E_i = bi.E
                A_j = bj.A_lagged
                p_j = 1 - (bj.E / max_E)
                b_j = A_j / max_A
                b_j_samples.append(b_j)
                # Skip boundary cases where the clamp doesn't apply
                if p_j <= 0 or p_j >= 1.0 or (1 - b_j * eta) <= 0:
                    continue
                num = gamma * E_i + p_j * (1 - b_j) * alpha * A_j
                den = p_j * (1 - b_j * eta)
                unclamped = num / den
                total_evals += 1
                if unclamped > bi.C + 1e-12:
                    clamp_hits += 1
                else:
                    nonclamp_hits += 1

    m.do_interest_rate = wrapped_do_ir
    m.simulate_full()
    m.finish()

    print("Phase 1 R1: C-clamp fraction + b_j distribution at baseline")
    print("  omega=0.5, eta=0.1, T=1000, seed=%d" % SEED)
    print()
    if total_evals > 0:
        clamp_frac = clamp_hits / total_evals
        print("  total non-boundary eq.6 evals: %d" % total_evals)
        print("  clamp hits (unclamped > C_i): %d (%.1f%%)" % (clamp_hits, 100 * clamp_frac))
        print("  non-clamp:                    %d (%.1f%%)" % (nonclamp_hits, 100 * (1 - clamp_frac)))
    else:
        print("  no non-boundary evals?")
    print()
    if b_j_samples:
        b = np.array(b_j_samples)
        print("  b_j distribution across all (j, t):")
        print("    n samples = %d" % len(b))
        print("    mean      = %.3f" % b.mean())
        print("    median    = %.3f" % np.median(b))
        print("    p90       = %.3f" % np.percentile(b, 90))
        print("    p99       = %.3f" % np.percentile(b, 99))
        print("    max       = %.3f" % b.max())
        # Concentration: what fraction of total b_j mass is in the top 10% of banks?
        bsort = np.sort(b)[::-1]
        top10_idx = int(0.1 * len(bsort))
        if top10_idx > 0:
            top10_share = bsort[:top10_idx].sum() / bsort.sum()
            print("    top 10%% of (j,t) holds %.1f%% of total b_j mass" % (100 * top10_share))


if __name__ == "__main__":
    main()
