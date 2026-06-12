"""Phase Y: intermediate omega + hetero, default mu=0.7, default gamma=0.10.

Tests whether hetero at lower omega (less aggressive (mu, omega) decoupling)
gives a different sweet spot than C3 (mu=0.6, omega=0.7).

omega in {0.55, 0.58, 0.60} x cv in {0.0, 0.5, 1.0}, default mu=0.7, default gamma=0.10
eta in {0, 0.1} social, 5 seeds = 90 sims, ~15 min.

Run AFTER Phase X completes. Helps decide if the (mu, omega) decoupling we did
for C3 is preferable to default-mu + intermediate-omega + hetero.

raw-id turnover counting.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import math
import csv
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS  = [26462, 26463, 26464, 26465, 26466]
WORKERS = 6
MU = 0.70                       # default
GAMMA = 0.10                    # default

OMEGAS = [0.55, 0.58, 0.60]
CVS    = [0.0, 0.5, 1.0]
ETAS   = [0.0, 0.1]


def run_one(args):
    omega, cv, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['equity_heterogeneity'] = (cv > 0)
    cfg['equity_cv'] = cv
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
    raw_ids = [b for b in bl if b >= 0]
    runs = []
    if raw_ids:
        prev = raw_ids[0]; rl = 1
        for k in raw_ids[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    avg_ten = sum(runs)/len(runs) if runs else 0
    blc = [s.best_lender_clients[t] for t in range(T) if s.best_lender_clients[t] >= 0]
    avg_cli = sum(blc)/len(blc) if blc else 0
    return {
        'omega':    omega, 'cv': cv, 'eta': eta, 'seed': seed,
        'total_bk': int(sum(s.bankruptcy[:T])),
        'contagion': int(sum(s.bankruptcies_contagion[:T])),
        'avg_cli':  round(avg_cli, 2),
        'avg_ten':  round(avg_ten, 2),
        'max_ten':  max_ten,
        'turnovers': max(0, len(runs) - 1),
    }


def msd(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x)) and not math.isinf(float(x))]
    if not xs: return 'n/a'
    if len(xs) < 2: return f'{xs[0]:.2f}'
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    s = math.sqrt(var)
    if m > 100: return f'{m:.0f}+/-{s:.0f}'
    if m > 1:   return f'{m:.2f}+/-{s:.2f}'
    return f'{m:.4f}+/-{s:.4f}'


def main():
    jobs = [(om, cv, e, s) for om in OMEGAS for cv in CVS for e in ETAS for s in SEEDS]
    print(f'Phase Y: intermediate omega + hetero ({len(jobs)} sims, {WORKERS} workers)')
    print(f'  fixed: mu={MU}, gamma={GAMMA}')
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_hetero_groupY_raw.csv')
    keys = ['omega', 'cv', 'eta', 'seed', 'total_bk', 'contagion',
            'avg_cli', 'avg_ten', 'max_ten', 'turnovers']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f'  saved raw rows -> {out_csv}')

    print()
    print(f'=== 5-seed aggregates per (omega, cv) cell at mu={MU}, gamma={GAMMA} (raw-id turnovers) ===')
    print(f'{"omega":>5} | {"cv":>5} | {"bk_eta0":>9} | {"bk_eta01":>9} | {"delta":>7} | {"claim3":>7} | '
          f'{"contagion":>11} | {"avg_cli":>11} | {"avg_ten":>11} | {"max_ten":>11} | {"turnovers":>11}')
    print('-' * 145)
    for om in OMEGAS:
        for cv in CVS:
            c0 = [r for r in rows if r['omega']==om and r['cv']==cv and r['eta']==0.0]
            c1 = [r for r in rows if r['omega']==om and r['cv']==cv and r['eta']==0.1]
            bk0 = sum(x['total_bk'] for x in c0) / len(c0) if c0 else 0
            bk1 = sum(x['total_bk'] for x in c1) / len(c1) if c1 else 0
            delta = bk1 - bk0
            verdict = 'YES' if delta < 0 else 'NO'
            cv_lbl = f'{cv:.2f}' if cv > 0 else '0.00'
            print(f'{om:>5.2f} | {cv_lbl:>5} | {bk0:>9.0f} | {bk1:>9.0f} | {delta:>+7.0f} | {verdict:>7} | '
                  f'{msd([x["contagion"] for x in c1]):>11} | '
                  f'{msd([x["avg_cli"] for x in c1]):>11} | '
                  f'{msd([x["avg_ten"] for x in c1]):>11} | '
                  f'{msd([x["max_ten"] for x in c1]):>11} | '
                  f'{msd([x["turnovers"] for x in c1]):>11}')
        print()


if __name__ == '__main__':
    main()
