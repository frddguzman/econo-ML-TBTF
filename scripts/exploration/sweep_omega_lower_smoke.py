"""Lower-omega ω-sweep smoke — 2 stages, sequential.

STAGE 1 (90 sims, ~8 min): ω-sweep at the existing tier+bilateral cells we already
have Brini diagrams for. Tests if lower ω destabilizes the lock-in dynamic.
  - bsl + tier_init + bilateral + γ=0.10 + mult=1.75 + K=3
  - C3  + tier_init + bilateral + γ=0.12 + mult=1.75 + K=3

STAGE 2 (360 sims, ~30 min): 8-config matrix WITHOUT tier_init at γ=0.10, cv=0.7.
  - 2 cands × 2 hetero × 2 basis = 8 configs
  - ω-sweep, all at γ=0.10, hetero uses cv=0.7

ω-grids (0.025 step, 5 values):
  - bsl-style (μ=0.7): {0.40, 0.425, 0.45, 0.475, 0.50}
  - C3-style  (μ=0.6): {0.60, 0.625, 0.65, 0.675, 0.70}

Common: 3 seeds {26462, 26463, 26464}, 3 etas {0, 0.1, 0.85}, social_tax regime.
Total: 90 + 360 = 450 sims, ~38 min wall (6 workers, sequential stages).

Output: sweep_omega_lower_smoke_raw.csv (full per-sim data, checkpointed every 50).
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

SEEDS = [26462, 26463, 26464]
ETAS = [0.0, 0.1, 0.85]
WORKERS = 6

OMEGA_BSL = [0.40, 0.425, 0.45, 0.475, 0.50]
OMEGA_C3  = [0.60, 0.625, 0.65, 0.675, 0.70]

CHECKPOINT_EVERY = 50

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'sweep_omega_lower_smoke_raw.csv')

KEYS = ['stage', 'cand', 'mu', 'omega', 'gamma', 'hetero', 'cv', 'basis',
        'tier_init', 'n_big', 'E_big_mult',
        'eta', 'seed', 'fiscal_regime',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness',
        'rotation_count', 'big_deaths']


def _sum_int(stats_obj, name, T):
    arr = getattr(stats_obj, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    """Universal runner — args is a dict with all per-sim parameters."""
    stage = args['stage']
    cand = args['cand']
    mu = args['mu']
    omega = args['omega']
    gamma = args['gamma']
    hetero = args['hetero']
    cv = args['cv']
    basis = args['basis']
    tier_init = args['tier_init']
    n_big = args['n_big']
    E_big_mult = args['E_big_mult']
    eta = args['eta']
    seed = args['seed']

    cfg = ddr.make_config(basis=basis, omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    if hetero:
        cfg['equity_heterogeneity'] = True
        cfg['equity_cv'] = cv
    if tier_init:
        cfg['tier_init'] = True
        cfg['n_big'] = n_big
        cfg['E_big_multiplier'] = E_big_mult

    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics

    # Hub stats
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

    # Tier-init diagnostics (zero if tier_init not enabled — defaults set in Model.__init__)
    rotation_count = (len(m.tier_init_ever_big_ids) - n_big) if tier_init else 0
    big_deaths = m.big_bank_death_count if tier_init else 0

    return {
        'stage': stage, 'cand': cand, 'mu': mu, 'omega': omega, 'gamma': gamma,
        'hetero': hetero, 'cv': cv if hetero else 0.0, 'basis': basis,
        'tier_init': tier_init, 'n_big': n_big if tier_init else 0,
        'E_big_mult': E_big_mult if tier_init else 0.0,
        'eta': eta, 'seed': seed, 'fiscal_regime': 'socialized_tax',
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
        'rotation_count': rotation_count,
        'big_deaths':     big_deaths,
    }


def build_stage1_jobs():
    """Stage 1: 2 tier+bilateral cells × 5 ω × 3 η × 3 seeds = 90 sims."""
    jobs = []
    # bsl: μ=0.7, γ=0.10
    for omega in OMEGA_BSL:
        for eta in ETAS:
            for seed in SEEDS:
                jobs.append({
                    'stage': 'S1', 'cand': 'bsl', 'mu': 0.70, 'omega': omega, 'gamma': 0.10,
                    'hetero': False, 'cv': 0.0, 'basis': 'bilateral',
                    'tier_init': True, 'n_big': 3, 'E_big_mult': 1.75,
                    'eta': eta, 'seed': seed,
                })
    # C3: μ=0.6, γ=0.12 (matches Brini diagrams)
    for omega in OMEGA_C3:
        for eta in ETAS:
            for seed in SEEDS:
                jobs.append({
                    'stage': 'S1', 'cand': 'c3', 'mu': 0.60, 'omega': omega, 'gamma': 0.12,
                    'hetero': False, 'cv': 0.0, 'basis': 'bilateral',
                    'tier_init': True, 'n_big': 3, 'E_big_mult': 1.75,
                    'eta': eta, 'seed': seed,
                })
    return jobs


def build_stage2_jobs():
    """Stage 2: 8 configs × 5 ω × 3 η × 3 seeds = 360 sims (no tier_init, γ=0.10, cv=0.7 if hetero)."""
    jobs = []
    cand_specs = [
        ('bsl', 0.70, 0.10, OMEGA_BSL),
        ('c3',  0.60, 0.10, OMEGA_C3),
    ]
    for cand, mu, gamma, omega_grid in cand_specs:
        for hetero in (False, True):
            for basis in ('equity', 'bilateral'):
                for omega in omega_grid:
                    for eta in ETAS:
                        for seed in SEEDS:
                            jobs.append({
                                'stage': 'S2', 'cand': cand, 'mu': mu, 'omega': omega, 'gamma': gamma,
                                'hetero': hetero, 'cv': 0.7 if hetero else 0.0, 'basis': basis,
                                'tier_init': False, 'n_big': 0, 'E_big_mult': 0.0,
                                'eta': eta, 'seed': seed,
                            })
    return jobs


def write_checkpoint(rows, n_done, n_total):
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in KEYS})
    print(f'  [checkpoint @ {n_done}/{n_total}] wrote {len(rows)} rows', flush=True)


def m_s(vals):
    n = len(vals)
    if n == 0: return 0.0, 0.0
    m = sum(vals)/n
    if n < 2: return m, 0.0
    v = sum((x-m)**2 for x in vals)/(n-1)
    return m, math.sqrt(v)


def print_stage_summary(rows, stage_name):
    """Per-cell summary: mean ± std for total_bk/max_ten/avg_ten by (cand, hetero, basis, omega)."""
    print(f'\n=== {stage_name} per-cell summary (mean across 3 seeds, η at η*-cell-min) ===', flush=True)
    cells = {}
    for r in rows:
        key = (r['cand'], r['hetero'], r['basis'], r['omega'], r['gamma'])
        cells.setdefault(key, {}).setdefault(r['eta'], []).append(r)

    print(f'  {"cand":>4} {"hetero":>6} {"basis":>10} {"ω":>5} {"γ":>5} | {"η*":>4} {"bk@0":>6} {"bk*":>6} {"Δ":>7} | '
          f'{"max_ten*":>15} {"s/m":>5} | {"avg_ten*":>13} {"s/m":>5} | {"contag*":>8} {"fisc*":>7} | claim3', flush=True)
    print('  ' + '-' * 175, flush=True)
    for key in sorted(cells.keys()):
        cand, hetero, basis, omega, gamma = key
        by_eta = cells[key]
        if 0.0 not in by_eta: continue
        bk0_m = m_s([r['total_bk'] for r in by_eta[0.0]])[0]
        eta_means = {eta: m_s([r['total_bk'] for r in recs])[0] for eta, recs in by_eta.items()}
        eta_star = min(eta_means, key=eta_means.get)
        bk_star = eta_means[eta_star]
        delta = bk_star - bk0_m
        star_recs = by_eta[eta_star]
        mt_m, mt_s = m_s([r['max_ten'] for r in star_recs])
        av_m, av_s = m_s([r['avg_ten'] for r in star_recs])
        co_m, _ = m_s([r['contagion'] for r in star_recs])
        fi_m, _ = m_s([r['fiscal_deaths'] for r in star_recs])
        sm_max = mt_s/mt_m if mt_m else 0
        sm_avg = av_s/av_m if av_m else 0
        verdict = 'YES' if delta < 0 else 'NO'
        print(f'  {cand:>4} {str(hetero):>6} {basis:>10} {omega:>5.3f} {gamma:>5.2f} | '
              f'{eta_star:>4.2f} {bk0_m:>6.0f} {bk_star:>6.0f} {delta:>+7.0f} | '
              f'{mt_m:>5.0f}±{mt_s:>5.0f}        {sm_max:>5.2f} | '
              f'{av_m:>5.2f}±{av_s:>5.2f}    {sm_avg:>5.2f} | '
              f'{co_m:>8.0f} {fi_m:>7.0f} | {verdict}', flush=True)


def main():
    print(f'Lower-ω smoke (2-stage, sequential)', flush=True)
    print(f'  workers={WORKERS}, seeds={SEEDS}, etas={ETAS}', flush=True)
    print(f'  output: {OUT_CSV}', flush=True)

    rows = []

    # ----- STAGE 1 -----
    jobs1 = build_stage1_jobs()
    print(f'\n=== STAGE 1: tier+bilateral cells ===', flush=True)
    print(f'  {len(jobs1)} sims, ω-grid bsl: {OMEGA_BSL}, C3: {OMEGA_C3}', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs1), 1):
            rows.append(r)
            if i % CHECKPOINT_EVERY == 0 or i == len(jobs1):
                write_checkpoint(rows, i, len(jobs1) + 360)
    print(f'STAGE 1 done ({len(jobs1)} sims)', flush=True)
    print_stage_summary([r for r in rows if r['stage'] == 'S1'], 'STAGE 1')

    # ----- STAGE 2 -----
    jobs2 = build_stage2_jobs()
    print(f'\n=== STAGE 2: 8-config matrix without tier_init ===', flush=True)
    print(f'  {len(jobs2)} sims, γ=0.10, cv=0.7 when hetero=ON', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs2), 1):
            rows.append(r)
            if i % CHECKPOINT_EVERY == 0 or i == len(jobs2):
                write_checkpoint(rows, len(jobs1) + i, len(jobs1) + len(jobs2))
    print(f'STAGE 2 done ({len(jobs2)} sims)', flush=True)
    print_stage_summary([r for r in rows if r['stage'] == 'S2'], 'STAGE 2')

    print(f'\nTotal sims: {len(rows)} | output: {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
