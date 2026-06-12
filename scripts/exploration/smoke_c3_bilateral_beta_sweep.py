"""beta sweep at C3+tier_init+BILATERAL+gamma=0.12+mult=1.75.

Hypothesis: with bilateral fitness, the lock-in dynamic differs from equity fitness
(Polya-urn delayed lock-in vs immediate Boltzmann pile-on). beta-sensitivity could
differ — at lower beta, slow-rotation hub may not lock in within T=1000.

Spec:
  - C3 (mu=0.60, omega=0.70, gamma=0.12)
  - tier_init=True, fitness_basis='bilateral', mult=1.75, K=3
  - beta in {0..8}, eta in {0, 0.1, 0.85}, seeds 26462-26466
  - Total: 9 x 3 x 5 = 135 sims, ~12 min wall.
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
BETAS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ETAS = [0.0, 0.1, 0.85]
MULT = 1.75
N_BIG = 3
WORKERS = 6
HARD_STOP_THRESHOLD = 350


def run_one(args):
    beta, eta, seed = args
    cfg = ddr.make_config(basis='bilateral', omega=0.70, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.60
    cfg['gamma_capital'] = 0.12
    cfg['beta'] = beta
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
    avg_ten = sum(runs)/len(runs) if runs else 0
    import numpy as np
    return {
        'beta': beta, 'eta': eta, 'seed': seed,
        'max_ten': max_ten, 'avg_ten': avg_ten,
        'total_bk': int(np.nansum(m.statistics.bankruptcy[:T])),
        'rotation_count': len(m.tier_init_ever_big_ids) - N_BIG,
        'big_deaths': m.big_bank_death_count,
    }


def m_s(vals):
    n = len(vals)
    if n == 0: return 0, 0
    m = sum(vals)/n
    if n < 2: return m, 0
    v = sum((x-m)**2 for x in vals)/(n-1)
    return m, math.sqrt(v)


def main():
    jobs = [(b, e, s) for b in BETAS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'beta sweep at C3+tier_init+BILATERAL+gamma=0.12+mult={MULT}', flush=True)
    print(f'  betas: {BETAS}, etas: {ETAS}, seeds: {SEEDS}, K={N_BIG}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers, hard stop at max_ten > {HARD_STOP_THRESHOLD}', flush=True)
    print('', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f'  ... {i}/{n} done', flush=True)

    print('', flush=True)
    print('=== Per-(beta, eta) summary ===', flush=True)
    print(f'  {"beta":>4} | {"eta":>5} | {"n>350":>5} | {"max_ten":>13} {"s/m":>5} | {"avg_ten":>13} {"s/m":>5} | {"total_bk":>13} | {"big_deaths":>11} | {"rotation":>9}', flush=True)
    print('  ' + '-' * 130, flush=True)
    for beta in BETAS:
        for eta in ETAS:
            cells = [r for r in rows if r['beta']==beta and r['eta']==eta]
            mt = [r['max_ten'] for r in cells]
            av = [r['avg_ten'] for r in cells]
            tb = [r['total_bk'] for r in cells]
            bd = [r['big_deaths'] for r in cells]
            ro = [r['rotation_count'] for r in cells]
            mt_m, mt_s = m_s(mt); av_m, av_s = m_s(av); tb_m, tb_s = m_s(tb)
            bd_m, _ = m_s(bd); ro_m, _ = m_s(ro)
            sm_max = mt_s/mt_m if mt_m > 0 else 0
            sm_avg = av_s/av_m if av_m > 0 else 0
            n_over = sum(1 for v in mt if v > HARD_STOP_THRESHOLD)
            print(f'  {beta:>4} | {eta:>5.2f} | {n_over:>5} | {mt_m:>5.0f}±{mt_s:>6.0f} {sm_max:>5.2f} | {av_m:>5.2f}±{av_s:>6.2f} {sm_avg:>5.2f} | {tb_m:>5.0f}±{tb_s:>6.0f} | {bd_m:>11.0f} | {ro_m:>9.1f}', flush=True)
        print('  ' + '-' * 130, flush=True)

    print('', flush=True)
    print('=== Claim 3 verdict per beta (bk@eta=0 vs eta=0.1 vs eta=0.85, mean across 5 seeds) ===', flush=True)
    print(f'  {"beta":>4} | {"bk@e=0":>10} {"bk@e=0.1":>10} {"bk@e=0.85":>11} | {"delta(0.1)":>12} {"delta(0.85)":>13} | {"verdict":>10}', flush=True)
    print('  ' + '-' * 95, flush=True)
    for beta in BETAS:
        bks = {}
        for eta in ETAS:
            cells = [r for r in rows if r['beta']==beta and r['eta']==eta]
            bks[eta] = m_s([r['total_bk'] for r in cells])[0]
        d01 = bks[0.1] - bks[0.0]
        d085 = bks[0.85] - bks[0.0]
        if min(d01, d085) < 0:
            verdict = 'YES'
        else:
            verdict = 'NO'
        print(f'  {beta:>4} | {bks[0.0]:>10.0f} {bks[0.1]:>10.0f} {bks[0.85]:>11.0f} | {d01:>+12.0f} {d085:>+13.0f} | {verdict:>10}', flush=True)

    print('', flush=True)
    print('=== Per-cell tables (beta x eta) ===', flush=True)
    for beta in BETAS:
        print(f'\nbeta={beta}:', flush=True)
        for eta in ETAS:
            cells = sorted([r for r in rows if r['beta']==beta and r['eta']==eta], key=lambda x: x['seed'])
            mts = [int(r['max_ten']) for r in cells]
            avs = [round(r['avg_ten'], 1) for r in cells]
            tbs = [r['total_bk'] for r in cells]
            print(f'  eta={eta}: max_ten={mts}  avg_ten={avs}  total_bk={tbs}', flush=True)


if __name__ == '__main__':
    main()
