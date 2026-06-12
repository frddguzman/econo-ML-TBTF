"""Approach B (final delta from v6 §10): truncated lognormal hetero.

Spec:
  - 3 cands: bsl (mu=0.7, omega=0.50, gamma=0.10), w55 (mu=0.7, omega=0.55, gamma=0.10),
             C3+gamma=0.12 (mu=0.6, omega=0.70, gamma=0.12)
  - 3 cv: {0.5, 0.7, 1.0}
  - max_factor = 2.5, min_factor = 0.5  (truncate lognormal at 2.5*E_mean, floor at 0.5*E_mean)
  - 12 etas: full grid {0, 0.05, 0.1, 0.15, 0.2, 0.3, ..., 0.9}
  - 5 seeds: {26462-26466}
  - Total: 3 x 3 x 12 x 5 = 540 sims, ~45 min wall
  - Output: sweep_approach_b_raw.csv

Hypothesis: lognormal cv >= 0.7 produces star networks via extreme tail draws (E ~ 4-6x mean).
Truncating tails (cap E at 2.5x mean) prevents extreme outliers without changing the model's
core behavior. If this dissolves the star pattern -> potentially viable. If not -> structural
foreclosure complete.
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
CVS = [0.5, 0.7, 1.0]
MAX_FACTOR = 2.5
MIN_FACTOR = 0.5
WORKERS = 6

CANDS = [
    ('bsl', 0.70, 0.50, 0.10),
    ('w55', 0.70, 0.55, 0.10),
    ('c3',  0.60, 0.70, 0.12),
]

CHECKPOINT_EVERY = 100
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'sweep_approach_b_raw.csv')

KEYS = ['cand', 'mu', 'omega', 'gamma', 'cv', 'max_factor', 'min_factor',
        'eta', 'seed', 'fiscal_regime',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def run_one(args):
    cand, mu, omega, gamma, cv, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = cv
    cfg['equity_max_factor'] = MAX_FACTOR
    cfg['equity_min_factor'] = MIN_FACTOR
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
        'cand': cand, 'mu': mu, 'omega': omega, 'gamma': gamma, 'cv': cv,
        'max_factor': MAX_FACTOR, 'min_factor': MIN_FACTOR,
        'eta': eta, 'seed': seed, 'fiscal_regime': 'socialized_tax',
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


def write_checkpoint(rows, n_done, n_total):
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in KEYS})
    print(f'  [checkpoint @ {n_done}/{n_total}] wrote {len(rows)} rows', flush=True)


def m_s(vals):
    n = len(vals)
    if n == 0: return 0, 0
    m = sum(vals)/n
    if n < 2: return m, 0
    v = sum((x-m)**2 for x in vals)/(n-1)
    return m, math.sqrt(v)


def main():
    jobs = [(c[0], c[1], c[2], c[3], cv, e, s)
            for c in CANDS for cv in CVS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'Approach B (truncated lognormal cap={MAX_FACTOR}xE_mean, floor={MIN_FACTOR}xE_mean)', flush=True)
    print(f'  cands: {[c[0] for c in CANDS]}', flush=True)
    print(f'  cvs: {CVS}, etas: {ETAS}, seeds: {SEEDS}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers', flush=True)
    print('', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % CHECKPOINT_EVERY == 0 or i == n:
                write_checkpoint(rows, i, n)

    print('', flush=True)
    print('=== Per-(cand, cv) claim 3 verdict + hub stats at optimum eta ===', flush=True)
    print(f'  {"cand":>4} {"cv":>4} | {"eta*":>5} | {"bk@0":>8} {"bk*":>8} {"delta":>8} | {"max_ten":>13} {"s/m":>5} | {"avg_ten":>13} {"s/m":>5} | {"verdict":>10}', flush=True)
    print('  ' + '-' * 130, flush=True)
    candidate_winners = []
    for cand, mu, omega, gamma in CANDS:
        for cv in CVS:
            cells_by_eta = {}
            for eta in ETAS:
                cells = [r for r in rows if r['cand']==cand and abs(r['cv']-cv)<1e-6 and abs(r['eta']-eta)<1e-6]
                if not cells: continue
                cells_by_eta[eta] = cells
            if 0.0 not in cells_by_eta: continue
            bk0 = m_s([r['total_bk'] for r in cells_by_eta[0.0]])[0]
            eta_means = {eta: m_s([r['total_bk'] for r in recs])[0] for eta, recs in cells_by_eta.items()}
            eta_star = min(eta_means, key=eta_means.get)
            bk_star = eta_means[eta_star]
            delta = bk_star - bk0
            star_recs = cells_by_eta[eta_star]
            mt_m, mt_s = m_s([r['max_ten'] for r in star_recs])
            av_m, av_s = m_s([r['avg_ten'] for r in star_recs])
            sm_max = mt_s / mt_m if mt_m else 0
            sm_avg = av_s / av_m if av_m else 0
            verdict = 'YES' if delta < 0 else 'NO'
            print(f'  {cand:>4} {cv:>4.2f} | {eta_star:>5.2f} | {bk0:>8.0f} {bk_star:>8.0f} {delta:>+8.0f} | '
                  f'{mt_m:>5.0f}±{mt_s:>5.0f}    {sm_max:>5.2f} | {av_m:>5.2f}±{av_s:>5.2f}    {sm_avg:>5.2f} | {verdict:>10}', flush=True)
            if delta < 0 and 15 <= mt_m <= 70 and sm_max < 0.4:
                candidate_winners.append((cand, cv, eta_star, mt_m, av_m, delta))

    print('', flush=True)
    if candidate_winners:
        print(f'=== POSSIBLE WINNERS (max_ten in [15,70] AND s/m<0.4 AND claim 3 preserved) ===', flush=True)
        for w in candidate_winners:
            print(f'  cand={w[0]} cv={w[1]} eta*={w[2]} max_ten={w[3]:.0f} avg_ten={w[4]:.2f} delta={w[5]:+.0f}', flush=True)
    else:
        print(f'=== No candidate beats the (max_ten in [15,70] AND s/m<0.4) target ===', flush=True)
        print(f'    Truncated lognormal does not escape the star-or-no-hub dichotomy.', flush=True)
        print(f'    Comprehensive foreclosure complete: commit w58 cv=0 by elimination.', flush=True)


if __name__ == '__main__':
    main()
