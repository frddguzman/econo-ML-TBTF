"""tau_fund -> 0 crossover test at w58 fund regime.

Tests whether fund regime converges to no-tax behavior (interior min at eta=0.5)
as tau -> 0. We know:
  no-tax at w58:  interior min at eta=0.5 (Δ=-218)
  tau=0.00005 fund: monotone up (Δ=+350 at eta=0.1)
  tau=0.0001 fund: monotone up (Δ=+640 at eta=0.1)

So crossover from "interior-min" to "broken" is somewhere in tau in [0, 0.00005].

tau values: {0, 1e-6, 1e-5, 5e-5} x 10 etas x 5 seeds = 200 sims, ~17 min.

tau=0 means equity_heterogeneity off AND fund_levy_rate=0. At fund_levy=0 the fund
never accumulates; bailouts effectively don't happen because fund is empty. Should
match no-tax behavior closely.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import math
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS  = [26462, 26463, 26464, 26465, 26466]
WORKERS = 6
MU = 0.70
OMEGA = 0.58
GAMMA = 0.10
REGIME = 'resolution_fund'

TAUS = [0.0, 1e-6, 1e-5, 5e-5]
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _sum_int(stats_obj, name, T):
    arr = getattr(stats_obj, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    tau, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime=REGIME)
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['fund_levy_rate'] = tau
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
        'tau': tau, 'eta': eta, 'seed': seed,
        'total_bk':       _sum_int(s, 'bankruptcy', T),
        'shock':          _sum_int(s, 'bankruptcies_shock', T),
        'rationing':      _sum_int(s, 'bankruptcies_rationing', T),
        'repay':          _sum_int(s, 'bankruptcies_repay', T),
        'contagion':      _sum_int(s, 'bankruptcies_contagion', T),
        'fiscal_deaths':  _sum_int(s, 'bankruptcies_fiscal', T),
        'zombies':        _sum_int(s, 'fire_sale_survivors', T),
        'avg_cli':        round(avg_cli, 2),
        'avg_ten':        round(avg_ten, 2),
        'max_ten':        max_ten,
        'turnovers':      max(0, len(runs) - 1),
    }


def main():
    jobs = [(t, e, s) for t in TAUS for e in ETAS for s in SEEDS]
    print(f'tau -> 0 crossover at w58 fund regime ({len(jobs)} sims, {WORKERS} workers)')
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_w58_tau_low_raw.csv')
    keys = ['tau', 'eta', 'seed', 'total_bk', 'shock', 'rationing', 'repay',
            'contagion', 'fiscal_deaths', 'zombies',
            'avg_cli', 'avg_ten', 'max_ten', 'turnovers']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f'  saved raw rows -> {out_csv}')

    # Per-tau eta-curve
    print()
    print(f'=== Total bk by tau and eta (5-seed mean) ===')
    print(f'{"tau":>10} | ' + ' | '.join(f'{e:>6.1f}' for e in ETAS) + ' | min eta | min val | claim 3')
    print('-' * 145)
    for tau in TAUS:
        means = []
        for eta in ETAS:
            cell = [r['total_bk'] for r in rows if r['tau']==tau and r['eta']==eta]
            means.append(sum(cell)/len(cell) if cell else 0)
        min_idx = means.index(min(means))
        min_eta = ETAS[min_idx]
        delta = means[min_idx] - means[0]
        verdict = f'YES (eta*={min_eta}, d={delta:+.0f})' if delta < 0 else 'NO (eta=0 wins)'
        print(f'{tau:>10.6f} | ' + ' | '.join(f'{m:>6.0f}' for m in means) + f' | {min_eta:>6.1f} | {means[min_idx]:>7.0f} | {verdict}')

    # Compare with no-tax baseline (read existing thesis_lehman_w58.csv)
    print()
    print(f'=== Reference: no-tax at w58 (from thesis_lehman_w58.csv) ===')
    try:
        from pathlib import Path
        # No-tax reference CSV location. Override via TBTF_SIM_DIR env var.
        # Default: a sibling 'Simulations/' folder next to the repo root.
        nt_path = os.environ.get(
            'TBTF_W58_NOTAX_CSV',
            str(Path(__file__).resolve().parent.parent / 'Simulations' / 'thesis_lehman_w58.csv'),
        )
        nt_rows = []
        with open(nt_path) as f:
            for r in csv.DictReader(f):
                if r['fiscal_regime'] == 'none':
                    nt_rows.append(r)
        nt_means = []
        for eta in ETAS:
            cell = [int(r['total_bk']) for r in nt_rows if abs(float(r['eta']) - eta) < 0.001]
            nt_means.append(sum(cell)/len(cell) if cell else 0)
        min_idx = nt_means.index(min(nt_means))
        print(f'{"no-tax":>10}   | ' + ' | '.join(f'{m:>6.0f}' for m in nt_means) + f' | {ETAS[min_idx]:>6.1f} | {nt_means[min_idx]:>7.0f}')
    except FileNotFoundError:
        print('  (no-tax reference file not available)')


if __name__ == '__main__':
    main()
