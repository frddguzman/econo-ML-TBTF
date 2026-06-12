"""C3 full-grid smoke: same scope as bsl smoke (4 mults x 5 seeds x 3 etas = 60 sims),
at C3 (mu=0.60, omega=0.70, gamma=0.12) instead of bsl. Tests whether the ω-driven
rotation observed at C3 mult=1.5 generalizes across multipliers.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]
MULTS = [1.3, 1.5, 1.75, 2.0]
ETAS = [0.0, 0.1, 0.85]
N_BIG = 3
WORKERS = 6
HARD_STOP_THRESHOLD = 350


def run_one(args):
    mult, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.70, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.60
    cfg['gamma_capital'] = 0.12
    cfg['tier_init'] = True
    cfg['n_big'] = N_BIG
    cfg['E_big_multiplier'] = mult
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    initial_big_ids = sorted([b.id for b in m.banks if b.tier == 'big'])
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
    final_big_ids = sorted([b.id for b in m.banks if b.tier == 'big'])
    import numpy as np
    total_bk = int(np.nansum(m.statistics.bankruptcy[:T]))
    return {
        'mult': mult, 'eta': eta, 'seed': seed,
        'max_ten': max_ten, 'total_bk': total_bk,
        'initial_big_ids': initial_big_ids,
        'final_big_ids': final_big_ids,
        'rotation_count': len(m.tier_init_ever_big_ids) - N_BIG,
        'big_deaths': m.big_bank_death_count,
    }


def main():
    jobs = [(m, e, s) for m in MULTS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'C3 full-grid smoke (tier-init + empty-slot-fill rotation)', flush=True)
    print(f'  cand: C3 (mu=0.60, omega=0.70, gamma=0.12)', flush=True)
    print(f'  mults: {MULTS}, etas: {ETAS}, seeds: {SEEDS}', flush=True)
    print(f'  total: {n} sims, {WORKERS} workers, hard stop at max_ten > {HARD_STOP_THRESHOLD}', flush=True)
    print('', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 15 == 0:
                print(f'  ... {i}/{n} done', flush=True)

    print('', flush=True)
    print('=== C3 full smoke results (per multiplier; seeds in rows, etas in columns) ===', flush=True)
    over = []
    for mult in MULTS:
        print('', flush=True)
        print(f'mult={mult}:', flush=True)
        print(f'  {"seed":>5} | {"e=0.0":>7} {"e=0.1":>7} {"e=0.85":>7} | {"rot @ etas":>20} | {"big_dth @ etas":>22} | {"total_bk":>22}', flush=True)
        print(f'  {"-"*5}-+-{"-"*7}-{"-"*7}-{"-"*7}-+-{"-"*20}-+-{"-"*22}-+-{"-"*22}', flush=True)
        for seed in SEEDS:
            mt = []
            rot = []
            bd = []
            tb = []
            for eta in ETAS:
                r = next(x for x in rows if x['mult']==mult and x['eta']==eta and x['seed']==seed)
                mt.append(r['max_ten']); rot.append(r['rotation_count']); bd.append(r['big_deaths']); tb.append(r['total_bk'])
                if r['max_ten'] > HARD_STOP_THRESHOLD:
                    over.append((mult, seed, eta, r['max_ten']))
            mt_str = lambda v: f'**{v}**' if v > HARD_STOP_THRESHOLD else f'{v}'
            print(f'  {seed:>5} | {mt_str(mt[0]):>7} {mt_str(mt[1]):>7} {mt_str(mt[2]):>7} | {str(rot):>20} | {str(bd):>22} | {str(tb):>22}', flush=True)

    print('', flush=True)
    print('=== Summary ===', flush=True)
    print(f'  Total cells: {len(rows)}', flush=True)
    print(f'  Cells > {HARD_STOP_THRESHOLD}: {len(over)}', flush=True)
    print(f'  max_ten range: {min(r["max_ten"] for r in rows)} to {max(r["max_ten"] for r in rows)}', flush=True)
    print(f'  mean max_ten: {sum(r["max_ten"] for r in rows)/len(rows):.0f}', flush=True)
    print(f'  median max_ten: {sorted(r["max_ten"] for r in rows)[len(rows)//2]}', flush=True)
    print('', flush=True)

    # Per-mult summary
    for mult in MULTS:
        cells = [r for r in rows if r['mult']==mult]
        n_over = sum(1 for r in cells if r['max_ten'] > HARD_STOP_THRESHOLD)
        mean_mt = sum(r['max_ten'] for r in cells)/len(cells)
        max_mt = max(r['max_ten'] for r in cells)
        min_mt = min(r['max_ten'] for r in cells)
        print(f'  mult={mult}: {n_over}/{len(cells)} > 350; mean={mean_mt:.0f}; range [{min_mt}, {max_mt}]', flush=True)

    print('', flush=True)
    if over:
        print(f'Cells over threshold:', flush=True)
        for c in over:
            print(f'  mult={c[0]} seed={c[1]} eta={c[2]} -> max_ten={c[3]}', flush=True)


if __name__ == '__main__':
    main()
