"""C3 fine cv sweep — Goldilocks search in the untested cv ∈ (0, 0.7) gap.

Tests the hypothesis that the bimodality observed at C3 cv ∈ {0.7, 0.85, 1.0}
(champion-vs-churn seed-conditional outcomes) dissolves at moderate cv where
the lognormal tail is shorter and no bank gets initial equity 3-4x mean.

Spec:
  - 3 gamma values: {0.05, 0.10, 0.12}                (all C3 supervisor cells)
  - 6 cv values:    {0.30, 0.40, 0.50, 0.55, 0.60, 0.65}   (fills gap to existing cv=0.70 data)
  - 10 etas:        full 0.1 step from 0.0 to 0.9
  - 1 regime:       socialized_tax only               (decisive for canonical pick)
  - 5 seeds:        [26462-26466]
  - Total: 3 * 6 * 10 * 1 * 5 = 900 sims
  - 6 workers in one pool (respects 6-CPU constraint)
  - Estimated wall: ~75-85 min

Output: sweep_c3_cv_fine_raw.csv (full channel decomp + hub stats per sim).
Checkpoint save every 100 sims.

Acceptance criteria for a "Goldilocks" cell (printed in summary):
  1. mean(max_ten) at the optimum eta in [25, 70]
  2. zero seeds with max_ten > 200 at the optimum eta (no champion mode at all)
  3. std(max_ten) / mean(max_ten) < 0.35 at the optimum eta
  4. claim 3 preserved (mean delta < 0)

If a cell passes all 4, it qualifies for 15-seed validation. If multiple
cells pass, pick the one with deepest delta.

Schema is sweep_c3_hetero_full_eta_raw.csv compatible (same KEYS).

Anti-bug: runs only at gamma in {0.05, 0.10, 0.12}, cv NOT in the existing
overnight sweep (no overlap with sweep_c3_hetero_full_eta_raw.csv).

Note on tau: uses default tau=1e-4. Fund regime not tested here; if a winner
emerges, run the fund regime separately at the winner cell.
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
TAU_FUND = 1e-4

GAMMAS = [0.05, 0.10, 0.12]
CVS    = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65]
ETAS   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REGIMES = ['socialized_tax']

CHECKPOINT_EVERY = 100

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'sweep_c3_cv_fine_raw.csv')

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
    if not xs: return ('n/a', 0.0, 0.0)
    if len(xs) < 2:
        return (f'{xs[0]:.1f}', xs[0], 0.0)
    n = len(xs); m = sum(xs)/n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    s = math.sqrt(var)
    return (f'{m:.1f}+/-{s:.1f}', m, s)


def write_checkpoint(rows, n_done, n_total):
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in KEYS})
    print(f'  [checkpoint @ {n_done}/{n_total}] wrote {len(rows)} rows', flush=True)


def main():
    jobs = [(g, c, e, r, s) for g in GAMMAS for c in CVS for e in ETAS for r in REGIMES for s in SEEDS]
    n = len(jobs)
    print(f'C3 fine cv sweep (Goldilocks search)', flush=True)
    print(f'  cells: {len(GAMMAS)} gamma x {len(CVS)} cv = {len(GAMMAS)*len(CVS)}', flush=True)
    print(f'  etas: {len(ETAS)}, regimes: {len(REGIMES)}, seeds: {len(SEEDS)}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers', flush=True)
    print(f'  output: {OUT_CSV}', flush=True)
    print(f'  checkpoint: every {CHECKPOINT_EVERY} sims', flush=True)
    print('', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % CHECKPOINT_EVERY == 0 or i == n:
                write_checkpoint(rows, i, n)

    print('', flush=True)
    print(f'=== Per-cell claim 3 verdict (social, mean total_bk by eta) ===', flush=True)
    print(f'{"gamma":>6} | {"cv":>5} | ' + ' | '.join(f'{e:>6.1f}' for e in ETAS) +
          f' | {"eta*":>5} | {"bk*":>6} | {"bk@0":>6} | {"delta":>7} | {"claim3":>10}', flush=True)
    print('-' * 175, flush=True)
    for gamma in GAMMAS:
        for cv in CVS:
            vals = []
            for eta in ETAS:
                cell = [x['total_bk'] for x in rows if x['gamma']==gamma and x['cv']==cv
                        and x['fiscal_regime']=='socialized_tax' and abs(x['eta']-eta) < 1e-6]
                vals.append(sum(cell)/len(cell) if cell else 0)
            bk0 = vals[0]
            min_idx = vals.index(min(vals))
            min_eta = ETAS[min_idx]
            min_val = vals[min_idx]
            delta = min_val - bk0
            verdict = f'YES (e*={min_eta:.1f})' if delta < 0 else 'NO'
            print(f'{gamma:>6.2f} | {cv:>5.2f} | ' + ' | '.join(f'{m:>6.0f}' for m in vals) +
                  f' | {min_eta:>5.1f} | {min_val:>6.0f} | {bk0:>6.0f} | {delta:>+7.0f} | {verdict:>10}', flush=True)

    print('', flush=True)
    print(f'=== Goldilocks acceptance check (per cell, at the optimum eta) ===', flush=True)
    print(f'  Criteria: max_ten in [25,70], 0 seeds w/ max_ten>200, std/mean<0.35, delta<0', flush=True)
    print('', flush=True)
    print(f'{"gamma":>6} | {"cv":>5} | {"eta*":>5} | {"max_ten_per_seed":>30} | '
          f'{"mean":>6} | {"std":>5} | {"s/m":>5} | {"champ#":>6} | {"delta":>7} | {"GOLDILOCKS"}', flush=True)
    print('-' * 175, flush=True)

    winners = []
    for gamma in GAMMAS:
        for cv in CVS:
            # find optimum eta (lowest mean total_bk) for this cell
            vals_eta = {}
            for eta in ETAS:
                cell = [x['total_bk'] for x in rows if x['gamma']==gamma and x['cv']==cv
                        and x['fiscal_regime']=='socialized_tax' and abs(x['eta']-eta) < 1e-6]
                if cell: vals_eta[eta] = sum(cell)/len(cell)
            if not vals_eta: continue
            opt_eta = min(vals_eta, key=vals_eta.get)
            bk0 = vals_eta.get(0.0, 0)
            delta = vals_eta[opt_eta] - bk0

            # at optimum eta: per-seed max_ten
            opt_rows = sorted([x for x in rows if x['gamma']==gamma and x['cv']==cv
                              and x['fiscal_regime']=='socialized_tax' and abs(x['eta']-opt_eta) < 1e-6],
                             key=lambda r: r['seed'])
            mts = [r['max_ten'] for r in opt_rows]
            mts_str = '[' + ','.join(f'{int(m):>4}' for m in mts) + ']'
            n_mts = len(mts)
            mean_mt = sum(mts)/n_mts if n_mts else 0
            if n_mts >= 2:
                var_mt = sum((m - mean_mt) ** 2 for m in mts) / (n_mts - 1)
                std_mt = math.sqrt(var_mt)
            else:
                std_mt = 0
            sm_ratio = std_mt / mean_mt if mean_mt > 0 else float('inf')
            champ_count = sum(1 for m in mts if m > 200)

            c1 = 25 <= mean_mt <= 70
            c2 = champ_count == 0
            c3 = sm_ratio < 0.35
            c4 = delta < 0
            passes = c1 and c2 and c3 and c4
            verdict = 'PASS' if passes else f'fail({"".join(["1" if x else "." for x in [c1,c2,c3,c4]])})'
            if passes:
                winners.append((gamma, cv, opt_eta, mean_mt, std_mt, sm_ratio, champ_count, delta))

            print(f'{gamma:>6.2f} | {cv:>5.2f} | {opt_eta:>5.1f} | {mts_str:>30} | '
                  f'{mean_mt:>6.1f} | {std_mt:>5.1f} | {sm_ratio:>5.2f} | {champ_count:>6} | '
                  f'{delta:>+7.0f} | {verdict}', flush=True)

    print('', flush=True)
    if winners:
        print(f'=== {len(winners)} GOLDILOCKS WINNER(S) ===', flush=True)
        # rank by deepest delta
        winners.sort(key=lambda w: w[7])
        for w in winners:
            g, cv, eta, m, s, sm, ch, d = w
            print(f'  gamma={g}, cv={cv}, eta*={eta} -- max_ten {m:.0f}+/-{s:.0f} '
                  f'(s/m={sm:.2f}), champ_count={ch}, delta={d:+.0f}', flush=True)
        g, cv, eta, m, s, sm, ch, d = winners[0]
        print(f'', flush=True)
        print(f'TOP RECOMMENDATION: gamma={g}, cv={cv}, eta*={eta}', flush=True)
        print(f'  -> next step: 15-seed validation at this cell', flush=True)
    else:
        print(f'=== NO GOLDILOCKS WINNER ===', flush=True)
        print(f'  Recommendation: commit to w58 cv=0 by elimination.', flush=True)
        print(f'  Bimodality is structural at C3 hetero (any cv > 0).', flush=True)


if __name__ == '__main__':
    main()
