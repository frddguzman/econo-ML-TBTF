"""C3 hetero full thesis sweep — phase 1 followup to Phase 0.6 step 3.

Tests whether the C3 + cv hetero cells that survived deterministic 5-seed
screening (claim 3 preserved at eta in {0, 0.1} only) hold up under full
eta-grid AND across all 3 fiscal regimes. Also gives the "broken" cells
(cv=0.7, 0.85 at gamma 0.10/0.12) a fair shot — maybe the optimum just
shifted to higher eta.

Spec:
  - 3 gamma values: {0.05, 0.10, 0.12}
  - 3 cv values (hetero only): {0.70, 0.85, 1.00}
  - 10 etas: full 0.1 step from 0.0 to 0.9
  - 3 regimes: none, socialized_tax, resolution_fund
  - 5 seeds: [26462-26466] (matches all prior thesis sweeps)
  - 9 cells x 10 etas x 3 regimes x 5 seeds = 1,350 sims
  - 6 workers in one pool (respects 6-CPU constraint)
  - ~110-130 min wall

Output: sweep_c3_hetero_full_eta_raw.csv with full channel decomp + hub stats.
Checkpoint save every 100 sims to survive any mid-run interruption.
Schema is thesis_lehman_*.csv-compatible PLUS the cv column and hub stats.

Note on fund regime: this sweep uses tau=1e-4 (default). If claim 3 fails in
the fund regime for an otherwise-promising cell, that's NOT necessarily a
structural rejection — the levy rate is tweakable per v5 section 14.2 (tau->0
crossover at w58 showed fund regime can preserve claim 3 at sub-realistic tau).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import csv
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS  = [26462, 26463, 26464, 26465, 26466]
WORKERS = 6
MU = 0.60
OMEGA = 0.70
RHO = 0.4
BASIS = 'equity'
INERTIA = 0.0
TAU_FUND = 1e-4   # default; fund regime can be re-tweaked separately if needed

GAMMAS = [0.05, 0.10, 0.12]
CVS    = [0.70, 0.85, 1.00]
ETAS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REGIMES = ['none', 'socialized_tax', 'resolution_fund']

CHECKPOINT_EVERY = 100   # save partial CSV every N completed sims

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'sweep_c3_hetero_full_eta_raw.csv')

# Schema matches thesis_lehman_*.csv plus cv + hub stats.
KEYS = ['seed', 'fiscal_regime', 'rho', 'eta', 'omega', 'mu', 'gamma',
        'basis', 'inertia', 'cv',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def _sum_int(stats_obj, name, T):
    arr = getattr(stats_obj, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    gamma, cv, eta, regime, seed = args
    cfg = ddr.make_config(basis=BASIS, omega=OMEGA, eta=eta, regime=regime)
    cfg['mu'] = MU
    cfg['gamma_capital'] = gamma
    cfg['equity_heterogeneity'] = (cv > 0)
    cfg['equity_cv'] = cv
    cfg['fund_levy_rate'] = TAU_FUND
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics

    # Hub stats (raw-id turnover counting, dashboard-consistent)
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
    blf = [s.best_lender_fitness[t] for t in range(T)
           if s.best_lender_fitness[t] is not None and s.best_lender_fitness[t] >= 0]
    avg_fitness = sum(blf)/len(blf) if blf else 0
    bill = float(np.nansum(s.bailout_bill[:T])) if hasattr(s, 'bailout_bill') else 0.0

    return {
        'seed': seed, 'fiscal_regime': regime,
        'rho': RHO, 'eta': eta, 'omega': OMEGA, 'mu': MU, 'gamma': gamma,
        'basis': BASIS, 'inertia': INERTIA, 'cv': cv,
        'total_bk':       _sum_int(s, 'bankruptcy', T),
        'shock':          _sum_int(s, 'bankruptcies_shock', T),
        'rationing':      _sum_int(s, 'bankruptcies_rationing', T),
        'repay':          _sum_int(s, 'bankruptcies_repay', T),
        'contagion':      _sum_int(s, 'bankruptcies_contagion', T),
        'fiscal_deaths':  _sum_int(s, 'bankruptcies_fiscal', T),
        'zombies':        _sum_int(s, 'fire_sale_survivors', T),
        'bailout_bill':   round(bill, 2),
        'max_ten':        max_ten,
        'avg_ten':        round(avg_ten, 2),
        'turnovers':      max(0, len(runs) - 1),
        'avg_cli':        round(avg_cli, 2),
        'avg_fitness':    round(avg_fitness, 4),
    }


def msd(xs):
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x)) and not math.isinf(float(x))]
    if not xs: return 'n/a'
    if len(xs) < 2: return f'{xs[0]:.2f}'
    n = len(xs); m = sum(xs)/n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    s = math.sqrt(var)
    if m > 100: return f'{m:.0f}+/-{s:.0f}'
    if m > 1:   return f'{m:.2f}+/-{s:.2f}'
    return f'{m:.4f}+/-{s:.4f}'


def write_checkpoint(rows, n_done, n_total):
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in KEYS})
    print(f'  [checkpoint @ {n_done}/{n_total}] wrote {len(rows)} rows')


def main():
    jobs = [(g, c, e, r, s) for g in GAMMAS for c in CVS for e in ETAS for r in REGIMES for s in SEEDS]
    n = len(jobs)
    print(f'C3 hetero full eta sweep')
    print(f'  cells: {len(GAMMAS)} gamma x {len(CVS)} cv = {len(GAMMAS)*len(CVS)}')
    print(f'  etas: {len(ETAS)}, regimes: {len(REGIMES)}, seeds: {len(SEEDS)}')
    print(f'  total: {n} sims, {WORKERS} workers')
    print(f'  output: {OUT_CSV}')
    print(f'  checkpoint: every {CHECKPOINT_EVERY} sims')
    print()

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % CHECKPOINT_EVERY == 0 or i == n:
                write_checkpoint(rows, i, n)

    print()
    print(f'=== Final summary tables ===')
    print()

    # Per-cell claim 3 verdict per regime (mean total_bk by eta)
    for regime in REGIMES:
        print(f'-- regime: {regime} --')
        print(f'{"gamma":>6} | {"cv":>5} | ' + ' | '.join(f'{e:>6.1f}' for e in ETAS) +
              f' | {"min_eta":>7} | {"min_val":>8} | {"bk@0":>8} | {"delta":>7} | {"verdict":>10}')
        print('-' * 165)
        for gamma in GAMMAS:
            for cv in CVS:
                vals = []
                for eta in ETAS:
                    cell = [x['total_bk'] for x in rows if x['gamma']==gamma and x['cv']==cv
                            and x['fiscal_regime']==regime and abs(x['eta']-eta) < 1e-6]
                    vals.append(sum(cell)/len(cell) if cell else 0)
                bk0 = vals[0]
                min_idx = vals.index(min(vals))
                min_eta = ETAS[min_idx]
                min_val = vals[min_idx]
                delta = min_val - bk0
                verdict = f'YES (eta*={min_eta:.1f})' if delta < 0 else 'NO'
                print(f'{gamma:>6.2f} | {cv:>5.2f} | ' + ' | '.join(f'{m:>6.0f}' for m in vals) +
                      f' | {min_eta:>7.1f} | {min_val:>8.0f} | {bk0:>8.0f} | {delta:>+7.0f} | {verdict:>10}')
        print()

    # Channel decomp at eta=0.1 social per cell (engagement check)
    print(f'=== Channel decomp at eta=0.1 social (5-seed mean +/- std) ===')
    print(f'{"gamma":>6} | {"cv":>5} | {"shock":>11} | {"rationing":>11} | {"repay":>9} | '
          f'{"contagion":>11} | {"fiscal":>11} | {"zombies":>11} | {"bill":>11} | '
          f'{"max_ten":>11} | {"avg_cli":>11}')
    print('-' * 175)
    for gamma in GAMMAS:
        for cv in CVS:
            cells = [x for x in rows if x['gamma']==gamma and x['cv']==cv
                     and x['fiscal_regime']=='socialized_tax' and abs(x['eta']-0.1) < 1e-6]
            if not cells: continue
            print(f'{gamma:>6.2f} | {cv:>5.2f} | '
                  f'{msd([x["shock"] for x in cells]):>11} | '
                  f'{msd([x["rationing"] for x in cells]):>11} | '
                  f'{msd([x["repay"] for x in cells]):>9} | '
                  f'{msd([x["contagion"] for x in cells]):>11} | '
                  f'{msd([x["fiscal_deaths"] for x in cells]):>11} | '
                  f'{msd([x["zombies"] for x in cells]):>11} | '
                  f'{msd([x["bailout_bill"] for x in cells]):>11} | '
                  f'{msd([x["max_ten"] for x in cells]):>11} | '
                  f'{msd([x["avg_cli"] for x in cells]):>11}')


if __name__ == '__main__':
    main()
