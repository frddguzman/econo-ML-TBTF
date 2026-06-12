"""A/B test: does equity_heterogeneity=True meaningfully widen hub in-degree?

Runs A/B (hetero False vs True) across multiple (cand, seed) combinations to
test whether the structural concentration ceiling holds OR whether specific
parameter regimes break out of it.
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

# (label, basis, omega, seed)
COMBOS = [
    ('bsl    . seed 26474 (NEW)',     'equity',    0.50, 26474),
    ('bsl    . seed 26463 (OLD)',     'equity',    0.50, 26463),
    ('Cand A . seed 26474 (NEW)',     'equity',    0.53, 26474),
    ('Cand A . seed 26463 (OLD)',     'equity',    0.53, 26463),
    ('Cand B . seed 26474 (NEW)',     'bilateral', 0.52, 26474),
    ('Cand B . seed 26463 (OLD)',     'bilateral', 0.52, 26463),
    ('w55    . seed 26474 (NEW)',     'equity',    0.55, 26474),
    ('w55    . seed 26463 (OLD)',     'equity',    0.55, 26463),
]
ETA = 0.1
REGIME = 'socialized_tax'


def make_cfg(basis, omega, hetero):
    return dict(
        N=50, T=1000,
        mu=0.7, omega=omega,
        eta_bailout=ETA, rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=REGIME,
        fund_levy_rate=0.0001,
        fitness_basis=basis, fitness_inertia=0.0,
        equity_heterogeneity=hetero,
        equity_cv=0.5,
    )


def run_one(basis, omega, seed, hetero):
    cfg = make_cfg(basis, omega, hetero)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm('Boltzmann')
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    return m, T


def lender_max_tenure(model, T):
    s = model.statistics
    bl = list(s.best_lender[:T])
    bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def stats_for(model, T):
    deg = list(model.statistics.best_lender_clients[:T])
    deg_pos = [d for d in deg if d is not None and d >= 0]
    return {
        'total_bk':   int(sum(model.statistics.bankruptcy[:T])),
        'max_tenure': lender_max_tenure(model, T),
        'mean_deg':   statistics.mean(deg_pos) if deg_pos else 0,
        'med_deg':    statistics.median(deg_pos) if deg_pos else 0,
        'p95_deg':    statistics.quantiles(deg_pos, n=20)[18] if len(deg_pos) >= 20 else max(deg_pos, default=0),
        'max_deg':    max(deg_pos) if deg_pos else 0,
    }


def fmt(v): return f'{v:>6.2f}' if isinstance(v, float) else f'{v:>6}'

print(f'{"combo":<30} | {"variant":<8} | {"tot_bk":>7} {"max_ten":>7} {"mean_deg":>9} {"med":>4} {"p95":>4} {"max_deg":>7}')
print('-' * 110)
for label, basis, omega, seed in COMBOS:
    m_b, T_b = run_one(basis, omega, seed, hetero=False)
    s_b = stats_for(m_b, T_b)
    print(f'{label:<30} | {"hetero=F":<8} | {fmt(s_b["total_bk"])} {fmt(s_b["max_tenure"])} {fmt(s_b["mean_deg"])} {fmt(int(s_b["med_deg"]))} {fmt(int(s_b["p95_deg"]))} {fmt(s_b["max_deg"])}')
    m_h, T_h = run_one(basis, omega, seed, hetero=True)
    s_h = stats_for(m_h, T_h)
    print(f'{label:<30} | {"hetero=T":<8} | {fmt(s_h["total_bk"])} {fmt(s_h["max_tenure"])} {fmt(s_h["mean_deg"])} {fmt(int(s_h["med_deg"]))} {fmt(int(s_h["p95_deg"]))} {fmt(s_h["max_deg"])}')
    # Deltas
    d_meandeg = (s_h["mean_deg"] / s_b["mean_deg"] - 1) * 100 if s_b["mean_deg"] else 0
    d_maxten  = (s_h["max_tenure"] / s_b["max_tenure"] - 1) * 100 if s_b["max_tenure"] else 0
    d_maxdeg  = (s_h["max_deg"] / s_b["max_deg"] - 1) * 100 if s_b["max_deg"] else 0
    print(f'{"":>30} | {"DELTA":<8} | {"":>7} {d_maxten:+6.0f}% {d_meandeg:+8.0f}% {"":>4} {"":>4} {d_maxdeg:+6.0f}%')
    print('-' * 110)
