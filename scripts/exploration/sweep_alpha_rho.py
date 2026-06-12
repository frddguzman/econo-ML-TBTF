"""(alpha, rho) tuning at C3 calibration (mu=0.6, omega=0.7).

3x3 grid of (alpha_collateral, rho), varied around defaults:
  alpha ∈ {0.05, 0.07, 0.10}    (default 0.05; user notes "extremely sensitive")
  rho   ∈ {0.35, 0.40, 0.45}    (default 0.40)

For each cell: eta ∈ {0, 0.1} × 5 seeds (26462-26466), social regime.
Reports 5-seed mean ± std for total_bk, contagion, hub stats. Claim 3 verdict
per cell (delta = bk(0.1) - bk(0)).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import statistics as stats
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS  = [26462, 26463, 26464, 26465, 26466]
WORKERS = 6
MU = 0.60
OMEGA = 0.70

ALPHAS = [0.05, 0.07, 0.10]
RHOS   = [0.35, 0.40, 0.45]
ETAS   = [0.0, 0.1]

def run_one(args):
    alpha, rho, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime='socialized_tax')
    cfg['mu'] = MU
    cfg['alpha_collateral'] = alpha
    cfg['rho'] = rho
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
        'alpha': alpha, 'rho': rho, 'eta': eta, 'seed': seed,
        'total_bk':  int(sum(s.bankruptcy[:T])),
        'contagion': int(sum(s.bankruptcies_contagion[:T])),
        'avg_cli':   round(avg_cli, 2),
        'avg_ten':   round(avg_ten, 2),
        'max_ten':   max_ten,
        'turnovers': max(0, len(runs) - 1),
    }

def msd(xs):
    if len(xs) < 2: return f'{xs[0]:.1f}'
    m = stats.mean(xs); s = stats.stdev(xs)
    return f'{m:.0f}+/-{s:.0f}' if m > 100 else f'{m:.2f}+/-{s:.2f}'

def main():
    jobs = [(a, r, e, s)
            for a in ALPHAS for r in RHOS for e in ETAS for s in SEEDS]
    print(f'(alpha, rho) tuning at C3 (mu={MU}, omega={OMEGA}): {len(jobs)} sims, {WORKERS} workers')
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')

    print()
    print('=== 5-seed aggregates per (alpha, rho) cell ===')
    print(f'{"alpha":>5} | {"rho":>4} | {"bk_eta0":>9} | {"bk_eta01":>9} | {"delta":>7} | {"claim3":>7} | '
          f'{"avg_cli":>9} | {"avg_ten":>9} | {"max_ten":>9} | {"turnovers":>11}')
    print('-' * 130)
    for a in ALPHAS:
        for r in RHOS:
            cell0  = [row for row in rows if row['alpha']==a and row['rho']==r and row['eta']==0.0]
            cell01 = [row for row in rows if row['alpha']==a and row['rho']==r and row['eta']==0.1]
            bk0  = stats.mean([x['total_bk'] for x in cell0])
            bk01 = stats.mean([x['total_bk'] for x in cell01])
            delta = bk01 - bk0
            verdict = 'YES' if delta < 0 else 'NO'
            print(f'{a:>5.2f} | {r:>4.2f} | {bk0:>9.0f} | {bk01:>9.0f} | {delta:>+7.0f} | {verdict:>7} | '
                  f'{msd([x["avg_cli"] for x in cell01]):>9} | '
                  f'{msd([x["avg_ten"] for x in cell01]):>9} | '
                  f'{msd([x["max_ten"] for x in cell01]):>9} | '
                  f'{msd([x["turnovers"] for x in cell01]):>11}')
        print()

if __name__ == '__main__':
    main()
