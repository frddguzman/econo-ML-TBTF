"""Standalone A/B test for p_avg_ir spec change.

Runs Cand A · st · η=0.1 · seed 26474 in two configs:
  baseline: p_j = 1 - E_j/E_max (the default formula)
  p_avg:    p_j = 0.5 constant for all banks (the OG code's alternative)

Goal: confirm whether the 25% cash-cap clamp rate is driven by the dynamic
p_j formula, and see how hub-deg + tenure respond.

Doesn't touch any existing script.
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

SEED = 26474
BASIS = 'equity'
OMEGA = 0.53
ETA = 0.1
REGIME = 'socialized_tax'


def make_cfg(p_avg):
    return dict(
        N=50, T=1000,
        mu=0.7, omega=OMEGA,
        eta_bailout=ETA, rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=REGIME,
        fund_levy_rate=0.0001,
        fitness_basis=BASIS, fitness_inertia=0.0,
        equity_heterogeneity=False, equity_cv=0.5,
        p_avg_ir=p_avg,
    )


def _rle_max(keys):
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def _clamp_rate_for(model):
    """Recompute unclamped eq.6 cap and check what fraction would have exceeded
    C_i. Mirrors run_phase1.py's logic but inline. Sampled at one moment."""
    cnt_total = 0
    cnt_clamped = 0
    for i, bank_i in enumerate(model.banks):
        if bank_i.failed: continue
        for j, bank_j in enumerate(model.banks):
            if i == j or bank_j.failed: continue
            E_i = bank_i.E
            A_j = getattr(bank_j, 'A', bank_j.C + bank_j.L + bank_j.R)
            p_j = (1 - bank_j.prob_surviving)
            if p_j <= 0 or p_j >= 1: continue
            b_j = (getattr(bank_j, 'A_lagged', A_j) /
                   max((b.A_lagged for b in model.banks if not b.failed and getattr(b, 'A_lagged', 0) > 0),
                       default=1.0))
            denom = p_j * (1 - b_j * model.config.eta_bailout)
            if denom <= 0: continue
            num = (model.config.gamma_capital * E_i +
                   p_j * (1 - b_j) * model.config.alpha_collateral * A_j)
            unclamped = num / denom
            cnt_total += 1
            if unclamped > bank_i.C + 1e-12:
                cnt_clamped += 1
    return (cnt_clamped / cnt_total) if cnt_total else 0.0


def run_one(p_avg, label):
    cfg = make_cfg(p_avg)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm('Boltzmann')
    m.initialize(seed=SEED, generate_plots=False)

    # Sample clamp rate at 5 timepoints during the run
    clamp_samples = []
    for t in range(m.config.T):
        m.forward()
        if t in (100, 250, 500, 750, 999):
            clamp_samples.append(_clamp_rate_for(m))
    m.finish()

    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T]); bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    deg = list(s.best_lender_clients[:T])
    deg_pos = [d for d in deg if d is not None and d >= 0]
    return {
        'label':      label,
        'total_bk':   int(sum(s.bankruptcy[:T])),
        'max_tenure': _rle_max(keys),
        'mean_deg':   statistics.mean(deg_pos) if deg_pos else 0,
        'max_deg':    max(deg_pos) if deg_pos else 0,
        'p95_deg':    statistics.quantiles(deg_pos, n=20)[18] if len(deg_pos) >= 20 else max(deg_pos, default=0),
        'clamp_mean': statistics.mean(clamp_samples) if clamp_samples else 0,
    }


print(f'p_avg_ir A/B test on Cand A . st . eta=0.1 . seed {SEED}')
print('=' * 80)
print(f'{"label":<24} {"total_bk":>9} {"max_ten":>8} {"mean_deg":>9} {"max_deg":>8} {"p95":>5} {"clamp%":>7}')
print('-' * 80)
results = []
for p_avg, label in [(0.0, 'baseline (E/E_max)'), (0.5, 'p_avg_ir=0.5')]:
    r = run_one(p_avg, label)
    results.append(r)
    print(f'{r["label"]:<24} {r["total_bk"]:>9} {r["max_tenure"]:>8} '
          f'{r["mean_deg"]:>9.2f} {r["max_deg"]:>8} {int(r["p95_deg"]):>5} '
          f'{r["clamp_mean"]*100:>6.1f}%')

base = results[0]; pavg = results[1]
print('\n' + '=' * 80)
print('Comparison:')
print(f'  total_bk:    {base["total_bk"]:>6} -> {pavg["total_bk"]:>6} ({(pavg["total_bk"]/base["total_bk"]-1)*100:+.1f}%)')
print(f'  max_tenure:  {base["max_tenure"]:>6} -> {pavg["max_tenure"]:>6}')
print(f'  mean_deg:    {base["mean_deg"]:>6.2f} -> {pavg["mean_deg"]:>6.2f}')
print(f'  max_deg:     {base["max_deg"]:>6} -> {pavg["max_deg"]:>6}')
print(f'  clamp rate:  {base["clamp_mean"]*100:>5.1f}% -> {pavg["clamp_mean"]*100:>5.1f}%')
