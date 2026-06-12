"""Standalone Preferential vs Boltzmann test on Cand A.

No commitment — just see what happens with Preferential (BA topology + cash
boost) at the same Cand A parameters. Compare:
  baseline: Boltzmann (current model)
  pref:     Preferential m=3 (BA scale-free at init, guru gets 3x cash)

Track:
- total_bk, max_tenure, in-degree distribution (same as before)
- guru_id over time (the BA-highest-degree bank at init)
- guru_fitness over time (equity ratio of whoever currently holds the guru position)
- whether the original guru fails / gets replaced
- max in-degree per period
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


def make_cfg():
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
    )


def _rle_max(keys):
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def run_one(algorithm, m_param, label):
    cfg = make_cfg()
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    if m_param is not None:
        m.config.lender_change = lc.determine_algorithm(algorithm, m=m_param)
    else:
        m.config.lender_change = lc.determine_algorithm(algorithm)
    m.initialize(seed=SEED, generate_plots=False)

    # Capture initial state
    initial_guru = getattr(m.config.lender_change, 'guru', None)
    initial_C_max = max(b.C for b in m.banks)
    initial_E_max = max(b.E for b in m.banks)
    bank_id_at_max_C = next(b.id for b in m.banks if b.C == initial_C_max)

    # Track guru over time
    guru_id_trace = []
    guru_fitness_trace = []
    guru_failed_periods = []

    for t in range(m.config.T):
        m.forward()
        # Hub at this period (could differ from initial guru)
        hub_id = m.statistics.best_lender[t] if t < len(m.statistics.best_lender) else -1
        hub_fitness = (m.banks[hub_id].mu if 0 <= hub_id < len(m.banks) and not m.banks[hub_id].failed else 0)
        guru_id_trace.append(int(hub_id))
        guru_fitness_trace.append(float(hub_fitness))
        # Did the initial guru fail?
        if initial_guru is not None and initial_guru < len(m.banks) and m.banks[initial_guru].failed:
            guru_failed_periods.append(t)

    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T]); bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    deg = list(s.best_lender_clients[:T])
    deg_pos = [d for d in deg if d is not None and d >= 0]

    return {
        'label':         label,
        'initial_guru':  initial_guru,
        'initial_C_max': initial_C_max,
        'initial_E_max': initial_E_max,
        'bank_id_at_max_C': bank_id_at_max_C,
        'total_bk':      int(sum(s.bankruptcy[:T])),
        'max_tenure':    _rle_max(keys),
        'mean_deg':      statistics.mean(deg_pos) if deg_pos else 0,
        'max_deg':       max(deg_pos) if deg_pos else 0,
        'p95_deg':       statistics.quantiles(deg_pos, n=20)[18] if len(deg_pos) >= 20 else max(deg_pos, default=0),
        'guru_failed_at': guru_failed_periods[0] if guru_failed_periods else None,
        'unique_hubs':   len(set(g for g in guru_id_trace if g >= 0)),
        'time_at_initial_guru': sum(1 for g in guru_id_trace if g == initial_guru) if initial_guru else 0,
    }


print(f'Preferential vs Boltzmann test on Cand A . st . eta=0.1 . seed {SEED}')
print('=' * 90)
print(f'{"label":<25} {"tot_bk":>7} {"max_t":>6} {"mean_d":>7} {"max_d":>6} {"p95":>4} {"guru":>5} {"C_max":>7} {"E_max":>6}')
print('-' * 90)
results = []
for algo, m, label in [('Boltzmann', None, 'Boltzmann (baseline)'),
                        ('Preferential', 3, 'Preferential m=3'),
                        ('Preferential', 5, 'Preferential m=5')]:
    r = run_one(algo, m, label)
    results.append(r)
    print(f'{r["label"]:<25} {r["total_bk"]:>7} {r["max_tenure"]:>6} '
          f'{r["mean_deg"]:>7.2f} {r["max_deg"]:>6} {int(r["p95_deg"]):>4} '
          f'{str(r["initial_guru"]):>5} {r["initial_C_max"]:>7.1f} {r["initial_E_max"]:>6.2f}')

print('\n' + '=' * 90)
print('Detail:')
for r in results:
    print(f'\n  {r["label"]}:')
    print(f'    initial guru bank id:        {r["initial_guru"]}')
    print(f'    initial max-C bank (sanity): {r["bank_id_at_max_C"]}')
    print(f'    initial C of max bank:       {r["initial_C_max"]:.1f}')
    print(f'    initial E of max bank:       {r["initial_E_max"]:.2f}')
    if r["initial_guru"] is not None:
        print(f'    periods initial guru is hub: {r["time_at_initial_guru"]} / 1000')
    print(f'    unique hub identities seen:  {r["unique_hubs"]}')
    if r["guru_failed_at"] is not None:
        print(f'    initial guru FAILED at t = {r["guru_failed_at"]}')
    else:
        print(f'    initial guru did not fail')
