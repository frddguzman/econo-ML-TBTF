"""β + fitness-differential spike test.

Hypothesis: Boltzmann concentration ratio = exp(β·Δμ). β is constant in our
runs, but Δμ (fitness-spread across active banks per period) is what determines
actual concentration. If Δμ is small, increasing β doesn't help. Test:

For each (cand, β):
- Run sim, capture max_tenure + best_lender_clients
- Capture per-period Δμ = max(bank.mu) - min(bank.mu) across active banks
- Report effective concentration ratio = exp(β · mean(Δμ))
"""
import os, sys, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

SEED = 26474
ETA = 0.1
REGIME = 'socialized_tax'

# (label, basis, omega)
CANDS = [
    ('Cand A (equity, w=0.53)',     'equity',    0.53),
    ('Cand B (bilateral, w=0.52)',  'bilateral', 0.52),
    ('w55    (equity, w=0.55)',     'equity',    0.55),
]
BETAS = [1, 5, 10, 20]


def make_cfg(basis, omega, beta):
    return dict(
        N=50, T=1000,
        mu=0.7, omega=omega,
        eta_bailout=ETA, rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=beta, alfa=0.1,
        fiscal_regime=REGIME,
        fund_levy_rate=0.0001,
        fitness_basis=basis, fitness_inertia=0.0,
        equity_heterogeneity=False,
        equity_cv=0.5,
    )


def _rle_max(keys):
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def run_one(basis, omega, beta):
    cfg = make_cfg(basis, omega, beta)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm('Boltzmann')
    m.initialize(seed=SEED, generate_plots=False)
    # We'll measure Δμ across active banks each period via a forward-loop hook
    delta_mus = []
    fitness_top = []
    fitness_min = []
    for t in range(m.config.T):
        m.forward()
        # Capture fitness spread among non-failed banks
        mus = [b.mu for b in m.banks if not b.failed and b.mu is not None]
        if len(mus) >= 2:
            delta_mus.append(max(mus) - min(mus))
            fitness_top.append(max(mus))
            fitness_min.append(min(mus))
    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T]); bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    deg = list(s.best_lender_clients[:T])
    deg_pos = [d for d in deg if d is not None and d >= 0]
    return {
        'basis':       basis,
        'omega':       omega,
        'beta':        beta,
        'total_bk':    int(sum(s.bankruptcy[:T])),
        'max_tenure':  _rle_max(keys),
        'mean_deg':    statistics.mean(deg_pos) if deg_pos else 0,
        'max_deg':     max(deg_pos) if deg_pos else 0,
        'mean_dmu':    statistics.mean(delta_mus) if delta_mus else 0,
        'med_dmu':     statistics.median(delta_mus) if delta_mus else 0,
        'mean_top':    statistics.mean(fitness_top) if fitness_top else 0,
        'mean_min':    statistics.mean(fitness_min) if fitness_min else 0,
    }


print('beta x cand x fitness-differential test')
print('=' * 110)
print(f'{"cand":<28} {"beta":>4} {"max_ten":>7} {"max_deg":>7} {"mean_dmu":>9} {"top_mu":>7} {"min_mu":>7} {"exp(b*dmu)":>11}')
print('-' * 110)
for label, basis, omega in CANDS:
    for b in BETAS:
        r = run_one(basis, omega, b)
        ratio = float(np.exp(b * r["mean_dmu"]))
        print(f'{label:<28} {b:>4} {r["max_tenure"]:>7} {r["max_deg"]:>7} '
              f'{r["mean_dmu"]:>9.4f} {r["mean_top"]:>7.3f} {r["mean_min"]:>7.3f} '
              f'{ratio:>11.2f}')
    print('-' * 110)
