"""Fine eta sweep at bsl + equity + hetero(cv=0.7) + omega=0.45 + gamma=0.10.

Tests if there's a stable regime structure across eta: at low eta bimodal,
at eta=0.1 all-lock-in (frozen), at high eta lock-in with TBTF-inflation churn.
Full eta grid + 5 seeds for cleaner picture.

Total: 12 etas x 5 seeds = 60 sims, ~5 min wall.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]
ETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WORKERS = 6

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'sweep_bsl_hetero_w45_eta_fine_raw.csv')

KEYS = ['eta', 'seed', 'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def run_one(args):
    eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.45, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = 0.7
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
    blf = [s.best_lender_fitness[t] for t in range(T)
           if s.best_lender_fitness[t] is not None and s.best_lender_fitness[t] >= 0]
    avg_fitness = sum(blf)/len(blf) if blf else 0
    bill = float(np.nansum(s.bailout_bill[:T])) if hasattr(s, 'bailout_bill') else 0.0
    def _sum(name):
        arr = getattr(s, name, None)
        if arr is None: return 0
        return int(np.nansum(arr[:T]))
    return {
        'eta': eta, 'seed': seed,
        'total_bk': _sum('bankruptcy'),
        'shock': _sum('bankruptcies_shock'),
        'rationing': _sum('bankruptcies_rationing'),
        'repay': _sum('bankruptcies_repay'),
        'contagion': _sum('bankruptcies_contagion'),
        'fiscal_deaths': _sum('bankruptcies_fiscal'),
        'zombies': _sum('fire_sale_survivors'),
        'bailout_bill': round(bill, 2),
        'max_ten': max_ten,
        'avg_ten': round(avg_ten, 2),
        'turnovers': max(0, len(runs) - 1),
        'avg_cli': round(avg_cli, 2),
        'avg_fitness': round(avg_fitness, 4),
    }


def m_s(vals):
    n = len(vals)
    if n == 0: return 0, 0
    m = sum(vals)/n
    if n < 2: return m, 0
    v = sum((x-m)**2 for x in vals)/(n-1)
    return m, math.sqrt(v)


def main():
    jobs = [(e, s) for e in ETAS for s in SEEDS]
    print(f'Fine-eta sweep at bsl + equity + hetero(0.7) + omega=0.45 + gamma=0.10', flush=True)
    print(f'  {len(jobs)} sims, {WORKERS} workers', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)
    print(f'Done {len(rows)} sims.', flush=True)

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in KEYS})
    print(f'CSV: {OUT_CSV}', flush=True)

    # Per-eta diagnostic (mean across 5 seeds)
    print('', flush=True)
    print(f'=== Per-eta means + per-seed max_ten ===', flush=True)
    print(f'  {"eta":>5} | {"total_bk":>13} | {"contag":>10} | {"fisc":>8} | {"max_ten":>13} {"s/m":>5} | {"avg_ten":>13} {"s/m":>5} | {"max_ten by seed":>40}', flush=True)
    print('  ' + '-' * 175, flush=True)
    for eta in ETAS:
        cells = sorted([r for r in rows if abs(r['eta']-eta) < 1e-9], key=lambda x: x['seed'])
        tb_m, tb_s = m_s([r['total_bk'] for r in cells])
        co_m, _ = m_s([r['contagion'] for r in cells])
        fi_m, _ = m_s([r['fiscal_deaths'] for r in cells])
        mt_m, mt_s = m_s([r['max_ten'] for r in cells])
        av_m, av_s = m_s([r['avg_ten'] for r in cells])
        sm_max = mt_s/mt_m if mt_m else 0
        sm_avg = av_s/av_m if av_m else 0
        mts = [int(r['max_ten']) for r in cells]
        print(f'  {eta:>5.2f} | {tb_m:>5.0f}±{tb_s:>6.0f}    | {co_m:>10.0f} | {fi_m:>8.0f} | {mt_m:>5.0f}±{mt_s:>6.0f} {sm_max:>5.2f} | {av_m:>5.2f}±{av_s:>5.2f}    {sm_avg:>5.2f} | {str(mts):>40}', flush=True)

    # Claim 3 verdict
    eta_means = {}
    for eta in ETAS:
        cells = [r for r in rows if abs(r['eta']-eta) < 1e-9]
        eta_means[eta] = sum(r['total_bk'] for r in cells) / len(cells)
    bk0 = eta_means[0.0]
    eta_star = min(eta_means, key=eta_means.get)
    bk_star = eta_means[eta_star]
    delta = bk_star - bk0
    print('', flush=True)
    print(f'Claim 3: bk@0={bk0:.0f}, eta*={eta_star} bk*={bk_star:.0f}, delta={delta:+.0f}', flush=True)
    print(f'  verdict: {"PRESERVED" if delta < 0 else "BROKEN"}', flush=True)


if __name__ == '__main__':
    main()
