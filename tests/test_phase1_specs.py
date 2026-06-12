"""Phase 1 Step 8: verify the six fitness specs all run, total bankruptcies
in plausible range, bilateral state dicts populate, purge prevents dead-id leaks.
Single seed, T=1000.
"""
import numpy as np

import interbank
import interbank_lenderchange as lc


CONFIG_BASE = dict(
    N=50, T=1000,
    omega=0.50, mu=0.7,
    eta_bailout=0.1, rho=0.4,
    gamma_capital=0.10, alpha_collateral=0.05,
    beta=5, alfa=0.1,
    fiscal_regime="socialized_tax",
)
SEED = 26462

SPECS = [
    ("equity", 0.0),
    ("loan_book", 0.0),
    ("bilateral", 0.0),
    ("equity", 0.5),
    ("loan_book", 0.5),
    ("bilateral", 0.5),
]


def run(basis, inertia):
    cfg = dict(CONFIG_BASE)
    cfg["fitness_basis"] = basis
    cfg["fitness_inertia"] = inertia
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=SEED, generate_plots=False)
    m.simulate_full()
    m.finish()
    t = m.t
    total_bk = int(np.nansum(m.statistics.bankruptcy[:t]))
    contagion = int(np.nansum(m.statistics.bankruptcies_contagion[:t]))

    # Bilateral-specific diagnostics
    L_max_sys_present = hasattr(m, 'L_max_system')
    L_max_value = getattr(m, 'L_max_system', None)

    # Sample borrower dict size
    alive = [b for b in m.banks if not b.failed]
    sample_bank = alive[0] if alive else None
    dict_size = len(sample_bank.prev_smoothed_bilateral) if sample_bank else 0

    # Purge correctness check: every key in the sample dict should refer to
    # an alive bank id. (At end of sim, banks list reflects current state.)
    alive_ids = {b.id for b in alive}
    stale_keys = []
    if sample_bank:
        stale_keys = [k for k in sample_bank.prev_smoothed_bilateral
                      if k not in alive_ids]
    return {
        'T': t,
        'total_bk': total_bk,
        'contagion': contagion,
        'L_max_sys_present': L_max_sys_present,
        'L_max_value': L_max_value,
        'sample_dict_size': dict_size,
        'stale_keys_in_dict': len(stale_keys),
    }


def main():
    print("Phase 1 Step 8: spec verification (omega=0.5, eta=0.1, T=1000, seed=%d)" % SEED)
    print()
    print("%-12s %-9s %-9s %-9s %-15s %-12s %-12s" % (
        "basis", "inertia", "total_bk", "contagion", "L_max_system", "dict_size", "stale_keys",
    ))
    print("-" * 80)
    all_ok = True
    for basis, inertia in SPECS:
        r = run(basis, inertia)
        in_range = 10000 <= r['total_bk'] <= 25000
        bilateral_ok = (basis != "bilateral") or (r['L_max_sys_present'] and r['L_max_value'] > 0)
        purge_ok = r['stale_keys_in_dict'] == 0
        line_ok = in_range and bilateral_ok and purge_ok
        all_ok = all_ok and line_ok
        marker = "OK" if line_ok else "FAIL"
        L_str = ("%.2f" % r['L_max_value']) if r['L_max_value'] is not None else "n/a"
        print("%-12s %-9.2f %-9d %-9d %-15s %-12d %-12d %s" % (
            basis, inertia, r['total_bk'], r['contagion'],
            L_str, r['sample_dict_size'], r['stale_keys_in_dict'], marker,
        ))
    print()
    print("Acceptance:")
    print("  - total_bk in [10000, 25000]")
    print("  - bilateral specs: L_max_system populated and > 0")
    print("  - prev_smoothed_bilateral has no stale (dead) ids: passes purge correctness check")
    print("  Overall: %s" % ("ALL OK" if all_ok else "FAIL"))


if __name__ == "__main__":
    main()
