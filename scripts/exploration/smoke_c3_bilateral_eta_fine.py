"""Finer eta sweep at C3+tier_init+bilateral+mult=1.75 — verify the eta=0.1 dip
is real or a 3-eta-grid artefact.

Cell: C3 (mu=0.60, omega=0.70, gamma=0.12) + tier_init + fitness_basis='bilateral'
      + n_big=3 + E_big_multiplier=1.75
eta: 12 values {0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
seeds: 5

Total: 60 sims, ~5 min wall.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import math
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]
ETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MULT = 1.75
N_BIG = 3
WORKERS = 6


def run_one(args):
    eta, seed = args
    cfg = ddr.make_config(basis='bilateral', omega=0.70, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.60
    cfg['gamma_capital'] = 0.12
    cfg['tier_init'] = True
    cfg['n_big'] = N_BIG
    cfg['E_big_multiplier'] = MULT
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    bl = list(m.statistics.best_lender[:T])
    raw_ids = [b for b in bl if b >= 0]
    runs = []
    if raw_ids:
        prev = raw_ids[0]; rl = 1
        for k in raw_ids[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    import numpy as np
    return {
        'eta': eta, 'seed': seed,
        'max_ten': max_ten,
        'total_bk': int(np.nansum(m.statistics.bankruptcy[:T])),
        'rotation_count': len(m.tier_init_ever_big_ids) - N_BIG,
        'big_deaths': m.big_bank_death_count,
    }


def stats(values):
    n = len(values)
    if n == 0: return 0, 0
    mean = sum(values) / n
    if n < 2: return mean, 0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def main():
    jobs = [(e, s) for e in ETAS for s in SEEDS]
    print(f'C3 + tier_init + bilateral, mult=1.75, fine-eta sweep', flush=True)
    print(f'  total: {len(jobs)} sims, {WORKERS} workers', flush=True)
    print('', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)

    print('=== total_bk by (seed, eta) ===', flush=True)
    print(f'  {"seed":>5} | ' + ' '.join(f'{e:>6.2f}' for e in ETAS), flush=True)
    print('  ' + '-' * (8 + 7 * len(ETAS)), flush=True)
    for seed in SEEDS:
        vals = [next(r['total_bk'] for r in rows if r['seed']==seed and r['eta']==e) for e in ETAS]
        print(f'  {seed:>5} | ' + ' '.join(f'{v:>6}' for v in vals), flush=True)
    print('  ' + '-' * (8 + 7 * len(ETAS)), flush=True)
    means_tb = []
    for e in ETAS:
        vals = [r['total_bk'] for r in rows if r['eta']==e]
        m, s = stats(vals)
        means_tb.append(m)
    print(f'  {"mean":>5} | ' + ' '.join(f'{m:>6.0f}' for m in means_tb), flush=True)
    bk0 = means_tb[0]
    print(f'  {"d_vs_0":>5} | ' + ' '.join(f'{m-bk0:>+6.0f}' for m in means_tb), flush=True)

    print('', flush=True)
    print('=== max_ten by (seed, eta) ===', flush=True)
    print(f'  {"seed":>5} | ' + ' '.join(f'{e:>6.2f}' for e in ETAS), flush=True)
    print('  ' + '-' * (8 + 7 * len(ETAS)), flush=True)
    for seed in SEEDS:
        vals = [next(r['max_ten'] for r in rows if r['seed']==seed and r['eta']==e) for e in ETAS]
        print(f'  {seed:>5} | ' + ' '.join(f'{v:>6}' for v in vals), flush=True)
    print('  ' + '-' * (8 + 7 * len(ETAS)), flush=True)
    means_mt = []
    stds_mt = []
    for e in ETAS:
        vals = [r['max_ten'] for r in rows if r['eta']==e]
        m, s = stats(vals)
        means_mt.append(m)
        stds_mt.append(s)
    print(f'  {"mean":>5} | ' + ' '.join(f'{m:>6.0f}' for m in means_mt), flush=True)
    print(f'  {"std":>5} | ' + ' '.join(f'{s:>6.0f}' for s in stds_mt), flush=True)

    print('', flush=True)
    print('=== Claim 3 verdict ===', flush=True)
    min_idx = means_tb.index(min(means_tb))
    eta_star = ETAS[min_idx]
    bk_star = means_tb[min_idx]
    delta = bk_star - bk0
    print(f'  bk@eta=0: {bk0:.0f}', flush=True)
    print(f'  best eta: {eta_star} -> bk = {bk_star:.0f}', flush=True)
    print(f'  delta:    {delta:+.0f}', flush=True)
    if delta < 0:
        print(f'  VERDICT: claim 3 PRESERVED (eta*={eta_star})', flush=True)
    else:
        print(f'  VERDICT: claim 3 BROKEN (eta=0 wins)', flush=True)

    # Per-seed verdict
    print('', flush=True)
    print('=== Per-seed claim 3 ===', flush=True)
    for seed in SEEDS:
        vals = [next(r['total_bk'] for r in rows if r['seed']==seed and r['eta']==e) for e in ETAS]
        bk0_s = vals[0]
        min_v = min(vals)
        min_e = ETAS[vals.index(min_v)]
        d = min_v - bk0_s
        v = 'preserved' if d < 0 else 'broken'
        print(f'  {seed}: bk@0={bk0_s} best_eta={min_e} bk*={min_v} delta={d:+} ({v})', flush=True)

    # NEW — churn diagnostics: is low max_ten driven by big-bank deaths or by fitness rotation?
    print('', flush=True)
    print('=== big_deaths by (seed, eta) — high big_deaths = "churn by death" ===', flush=True)
    print(f'  {"seed":>5} | ' + ' '.join(f'{e:>6.2f}' for e in ETAS), flush=True)
    for seed in SEEDS:
        vals = [next(r['big_deaths'] for r in rows if r['seed']==seed and r['eta']==e) for e in ETAS]
        print(f'  {seed:>5} | ' + ' '.join(f'{v:>6}' for v in vals), flush=True)
    print('  ' + '-' * (8 + 7 * len(ETAS)), flush=True)
    means_bd = []
    for e in ETAS:
        vals = [r['big_deaths'] for r in rows if r['eta']==e]
        m, _ = stats(vals)
        means_bd.append(m)
    print(f'  {"mean":>5} | ' + ' '.join(f'{m:>6.0f}' for m in means_bd), flush=True)

    print('', flush=True)
    print('=== rotation_count by (seed, eta) — high rotation = many bank IDs ever held tier=big ===', flush=True)
    print(f'  {"seed":>5} | ' + ' '.join(f'{e:>6.2f}' for e in ETAS), flush=True)
    for seed in SEEDS:
        vals = [next(r['rotation_count'] for r in rows if r['seed']==seed and r['eta']==e) for e in ETAS]
        print(f'  {seed:>5} | ' + ' '.join(f'{v:>6}' for v in vals), flush=True)
    print('  ' + '-' * (8 + 7 * len(ETAS)), flush=True)
    means_rot = []
    for e in ETAS:
        vals = [r['rotation_count'] for r in rows if r['eta']==e]
        m, _ = stats(vals)
        means_rot.append(m)
    print(f'  {"mean":>5} | ' + ' '.join(f'{m:>6.1f}' for m in means_rot), flush=True)

    # Combined diagnostic: ratio big_deaths / max_ten — if high, lots of deaths per period of hub stability
    print('', flush=True)
    print('=== Diagnostic: per-eta means together ===', flush=True)
    print(f'  {"metric":>15} | ' + ' '.join(f'{e:>7.2f}' for e in ETAS), flush=True)
    print(f'  {"max_ten":>15} | ' + ' '.join(f'{m:>7.0f}' for m in means_mt), flush=True)
    print(f'  {"max_ten std":>15} | ' + ' '.join(f'{s:>7.0f}' for s in stds_mt), flush=True)
    print(f'  {"big_deaths":>15} | ' + ' '.join(f'{m:>7.0f}' for m in means_bd), flush=True)
    print(f'  {"rotation":>15} | ' + ' '.join(f'{m:>7.1f}' for m in means_rot), flush=True)
    print(f'  {"total_bk":>15} | ' + ' '.join(f'{m:>7.0f}' for m in means_tb), flush=True)


if __name__ == '__main__':
    main()
