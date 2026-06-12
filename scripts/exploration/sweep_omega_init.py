"""Multi-seed validation. Drop the anomalous seed 26474. Use thesis seed set
26462-26466 (5 seeds) and report mean ± std for each calibration.

Test 4 calibrations at eta=0 and eta=0.1, social regime:
  C1: bsl + median replacement (control, current dashboard baseline)
  C2: bsl + init replacement (the professor's directive — to retest)
  C3: H2-leading (mu=0.6, omega=0.7) + median replacement
  C4: H2-leading + init replacement

For each calibration: total_bk, contagion, hub max_ten, avg_ten, avg_cli,
turnovers — across 5 seeds, mean ± std reported. Claim 3 verdict per seed
and per-calibration aggregate.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')

import sys
import statistics as stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]

def run_one(mu, omega, eta, init_replace, seed):
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['reintroduce_with_median'] = not init_replace
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T])
    bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    runs = []
    if keys:
        prev = keys[0]; rl = 1
        for k in keys[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    avg_ten = sum(runs)/len(runs) if runs else 0
    blc = [s.best_lender_clients[t] for t in range(T) if s.best_lender_clients[t] >= 0]
    avg_cli = sum(blc)/len(blc) if blc else 0
    return {
        'total_bk':  int(sum(s.bankruptcy[:T])),
        'contagion': int(sum(s.bankruptcies_contagion[:T])),
        'avg_cli':   round(avg_cli, 2),
        'avg_ten':   round(avg_ten, 2),
        'max_ten':   max_ten,
        'turnovers': max(0, len(runs) - 1),
    }

def msd(xs):
    """Return 'mean ± std' as a string."""
    if len(xs) < 2: return f'{xs[0]:.1f}'
    m = stats.mean(xs); s = stats.stdev(xs)
    return f'{m:.0f} ± {s:.0f}' if m > 100 else f'{m:.2f} ± {s:.2f}'

CALIBRATIONS = [
    ('C1 bsl med   ', 0.70, 0.50, False),
    ('C2 bsl init  ', 0.70, 0.50, True),
    ('C3 H2 med    ', 0.60, 0.70, False),
    ('C4 H2 init   ', 0.60, 0.70, True),
]

print(f'Multi-seed validation: 5 seeds (NOT 26474), eta=0 vs eta=0.1 social.')
print('='*125)

results = {}
for label, mu, omega, init in CALIBRATIONS:
    print(f'\n--- {label} (mu={mu}, omega={omega}, init_replace={init}) ---')
    for eta in [0.0, 0.1]:
        rows = [run_one(mu, omega, eta, init, seed) for seed in SEEDS]
        results[(label.strip(), eta)] = rows
        print(f'  eta={eta}: total_bk={msd([r["total_bk"] for r in rows]):>15}  '
              f'contagion={msd([r["contagion"] for r in rows]):>13}  '
              f'avg_cli={msd([r["avg_cli"] for r in rows]):>13}  '
              f'avg_ten={msd([r["avg_ten"] for r in rows]):>13}  '
              f'max_ten={msd([r["max_ten"] for r in rows]):>13}  '
              f'turn={msd([r["turnovers"] for r in rows]):>10}')

print()
print('=== Claim 3 verdict (mean total_bk eta=0 vs eta=0.1 across 5 seeds) ===')
for label, mu, omega, init in CALIBRATIONS:
    name = label.strip()
    bk0 = stats.mean([r["total_bk"] for r in results[(name, 0.0)]])
    bk1 = stats.mean([r["total_bk"] for r in results[(name, 0.1)]])
    delta = bk1 - bk0
    verdict = 'PRESERVED' if delta < 0 else 'BROKEN'
    print(f'  {label}: bk(eta=0)={bk0:.0f}, bk(eta=0.1)={bk1:.0f}, delta={delta:+.0f} -> claim3 {verdict}')
