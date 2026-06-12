"""C3 diagnostic for tier-init lock-in (post-bsl-smoke-failure).

Tests if lock-in pattern observed at bsl is structural or cand-specific.
Hypothesis: at C3 (mu=0.6, omega=0.70 — higher shock variance) the hub bank
fails more often, reducing max_ten. Prediction: max_ten 400-600 range, still
above 350 threshold but lower than bsl's 700-900.

Spec: 1 cand (C3+gamma=0.12) x 1 mult (1.5) x 3 etas x 5 seeds = 15 sims, ~1 min wall.
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
ETAS = [0.0, 0.1, 0.85]
MULT = 1.5
N_BIG = 3
WORKERS = 6


def run_one(args):
    eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.70, eta=eta, regime='socialized_tax')
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
        'eta': eta, 'seed': seed,
        'max_ten': max_ten,
        'total_bk': total_bk,
        'initial_big_ids': initial_big_ids,
        'final_big_ids': final_big_ids,
        'rotation_count': len(m.tier_init_ever_big_ids) - N_BIG,
        'big_deaths': m.big_bank_death_count,
    }


def main():
    jobs = [(e, s) for e in ETAS for s in SEEDS]
    print(f'C3 diagnostic — tier-init at C3+gamma=0.12, mult={MULT}, K={N_BIG}', flush=True)
    print(f'  cand: C3 (mu=0.6, omega=0.70, gamma=0.12)', flush=True)
    print(f'  total: {len(jobs)} sims, {WORKERS} workers', flush=True)
    print('', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_one, jobs):
            rows.append(r)
    print(f'C3+gamma=0.12 mult={MULT}:', flush=True)
    print(f'  {"seed":>5} | {"eta":>5} | {"max_ten":>7} | {"total_bk":>8} | {"rot":>3} | {"big_dth":>7} | initial -> final', flush=True)
    print(f'  {"-"*5}-+-{"-"*5}-+-{"-"*7}-+-{"-"*8}-+-{"-"*3}-+-{"-"*7}-+-' + '-'*30, flush=True)
    for r in sorted(rows, key=lambda x: (x['seed'], x['eta'])):
        marker = '**' if r['max_ten'] > 350 else '  '
        print(f'  {r["seed"]:>5} | {r["eta"]:>5.2f} | {marker}{r["max_ten"]:>5}{marker} | {r["total_bk"]:>8} | {r["rotation_count"]:>3} | {r["big_deaths"]:>7} | {r["initial_big_ids"]} -> {r["final_big_ids"]}', flush=True)
    print('', flush=True)
    over_threshold = sum(1 for r in rows if r['max_ten'] > 350)
    print(f'  {over_threshold}/{len(rows)} cells over max_ten=350', flush=True)
    print(f'  max_ten range: {min(r["max_ten"] for r in rows)} to {max(r["max_ten"] for r in rows)}', flush=True)
    print(f'  mean max_ten: {sum(r["max_ten"] for r in rows)/len(rows):.0f}', flush=True)


if __name__ == '__main__':
    main()
