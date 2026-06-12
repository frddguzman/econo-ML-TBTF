"""Tier-init experiment — main sweep (5 cands x 3 E_big_mult x 12 etas x 5 seeds = 900 sims).

Tests the v6 §8 Option 4 hypothesis: persistent-tier-tag init with tier-aware replacement
addresses the §5.2.2 lock-in mechanism while preserving claim 3 + loud channels at louder
omega baselines.

Spec:
  - 5 cands: bsl (omega=0.50), A (0.53), w55 (0.55), w58 (0.58), C3 (mu=0.6, omega=0.70, gamma=0.12)
  - E_big_multiplier in {2.0, 3.0, 4.0}
  - n_big = 3 (fixed; FSB G-SIB anchor)
  - 12 etas: original 10-pt grid + diagnostic {0.05, 0.15}
  - regime: socialized_tax only
  - seeds: 26462-26466 (5)
  - Total: 5 x 3 x 12 x 5 = 900 sims, ~75 min wall (6 workers)

Output: sweep_tier_init_raw.csv, checkpointed every 100 sims.

Acceptance check (Goldilocks 3-way verdict per cell at the optimum eta):
  - PASS-CLEAN: max_ten in [15, 50] AND champion_count = 0 AND s/m < 0.35 AND delta < 0 AND Pyrrhic = False
  - PASS-PYRRHIC: same 4 base criteria but Pyrrhic = True (fiscal_fraction > 0.4)
  - FAIL: any base criterion fails
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

SEEDS = [26462, 26463, 26464, 26465, 26466]
WORKERS = 6
RHO = 0.4
BASIS = 'equity'
INERTIA = 0.0
TAU_FUND = 1e-4
N_BIG = 3

# (tag, mu, omega, gamma)
CANDS = [
    ('bsl', 0.70, 0.50, 0.10),
    ('a',   0.70, 0.53, 0.10),
    ('w55', 0.70, 0.55, 0.10),
    ('w58', 0.70, 0.58, 0.10),
    ('c3',  0.60, 0.70, 0.12),
]
E_BIG_MULTS = [2.0, 3.0, 4.0]
ETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CHECKPOINT_EVERY = 100

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'sweep_tier_init_raw.csv')

KEYS = ['seed', 'fiscal_regime', 'rho', 'eta', 'omega', 'mu', 'gamma',
        'basis', 'inertia', 'cand', 'n_big', 'E_big_mult',
        'total_bk', 'shock', 'rationing', 'repay', 'contagion',
        'fiscal_deaths', 'zombies', 'bailout_bill',
        'max_ten', 'avg_ten', 'turnovers', 'avg_cli', 'avg_fitness']


def _sum_int(stats_obj, name, T):
    arr = getattr(stats_obj, name, None)
    if arr is None: return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    cand_tag, mu, omega, gamma, e_big_mult, eta, seed = args
    cfg = ddr.make_config(basis=BASIS, omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    cfg['fund_levy_rate'] = TAU_FUND
    cfg['tier_init'] = True
    cfg['n_big'] = N_BIG
    cfg['E_big_multiplier'] = e_big_mult
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
        'seed': seed, 'fiscal_regime': 'socialized_tax',
        'rho': RHO, 'eta': eta, 'omega': omega, 'mu': mu, 'gamma': gamma,
        'basis': BASIS, 'inertia': INERTIA, 'cand': cand_tag,
        'n_big': N_BIG, 'E_big_mult': e_big_mult,
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


def write_checkpoint(rows, n_done, n_total):
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in KEYS})
    print(f'  [checkpoint @ {n_done}/{n_total}] wrote {len(rows)} rows', flush=True)


def main():
    jobs = [(c[0], c[1], c[2], c[3], m, e, s)
            for c in CANDS for m in E_BIG_MULTS for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'tier-init main sweep', flush=True)
    print(f'  cands: {len(CANDS)}, E_big_mults: {len(E_BIG_MULTS)}, etas: {len(ETAS)}, seeds: {len(SEEDS)}', flush=True)
    print(f'  cells: {len(CANDS)*len(E_BIG_MULTS)} = {len(CANDS)} cands x {len(E_BIG_MULTS)} mults', flush=True)
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
    print('=== Per-cell claim 3 verdict (social, mean total_bk by eta) ===', flush=True)
    eta_hdr = ' | '.join(f'{e:>5.2f}' for e in ETAS)
    print(f'{"cand":>5} | {"mult":>5} | {eta_hdr} | {"eta*":>5} | {"bk*":>6} | {"bk@0":>6} | {"delta":>7} | {"claim3":>10}', flush=True)
    print('-' * (40 + 8*len(ETAS)), flush=True)
    cell_summaries = []
    for cand in CANDS:
        cand_tag = cand[0]
        for mult in E_BIG_MULTS:
            vals = []
            for eta in ETAS:
                cell = [x['total_bk'] for x in rows
                        if x['cand']==cand_tag and abs(x['E_big_mult']-mult) < 1e-6
                        and abs(x['eta']-eta) < 1e-6]
                vals.append(sum(cell)/len(cell) if cell else 0)
            bk0 = vals[0]
            min_idx = vals.index(min(vals))
            min_eta = ETAS[min_idx]
            min_val = vals[min_idx]
            delta = min_val - bk0
            verdict = f'YES (e*={min_eta:.2f})' if delta < 0 else 'NO'
            print(f'{cand_tag:>5} | {mult:>5.1f} | ' + ' | '.join(f'{m:>5.0f}' for m in vals) +
                  f' | {min_eta:>5.2f} | {min_val:>6.0f} | {bk0:>6.0f} | {delta:>+7.0f} | {verdict:>10}', flush=True)
            cell_summaries.append({
                'cand': cand_tag, 'mult': mult,
                'opt_eta': min_eta, 'min_val': min_val, 'bk0': bk0, 'delta': delta,
            })

    print('', flush=True)
    print('=== Goldilocks acceptance check (per cell at optimum eta) ===', flush=True)
    print(f'  Criteria: max_ten in [15,50], 0 champions (max_ten>200), s/m<0.35, delta<0, Pyrrhic = (fiscal_fraction>0.4)', flush=True)
    print('', flush=True)
    print(f'{"cand":>5} | {"mult":>5} | {"eta*":>5} | {"max_ten_per_seed":>30} | '
          f'{"mean":>6} | {"std":>5} | {"s/m":>5} | {"chmp":>4} | '
          f'{"f_dth":>5} | {"f_frc":>5} | {"pyrr":>4} | {"delta":>7} | VERDICT', flush=True)
    print('-' * 175, flush=True)

    pass_clean, pass_pyrr, fail_cells = [], [], []
    for cs in cell_summaries:
        cand_tag = cs['cand']; mult = cs['mult']; opt_eta = cs['opt_eta']; delta = cs['delta']
        opt_rows = sorted([x for x in rows
                           if x['cand']==cand_tag and abs(x['E_big_mult']-mult) < 1e-6
                           and abs(x['eta']-opt_eta) < 1e-6],
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
        sm = std_mt / mean_mt if mean_mt > 0 else float('inf')
        champ = sum(1 for m in mts if m > 200)

        fiscal_at_opt = [r['fiscal_deaths'] for r in opt_rows]
        total_at_opt = [r['total_bk'] for r in opt_rows]
        mean_fiscal = sum(fiscal_at_opt)/n_mts if n_mts else 0
        mean_total = sum(total_at_opt)/n_mts if n_mts else 1
        fiscal_frac = mean_fiscal / mean_total if mean_total > 0 else 0
        pyrr = fiscal_frac > 0.4

        c1 = 15 <= mean_mt <= 50
        c2 = champ == 0
        c3 = sm < 0.35
        c4 = delta < 0
        if c1 and c2 and c3 and c4:
            if not pyrr:
                verdict = 'PASS-CLEAN'
                pass_clean.append((cand_tag, mult, opt_eta, mean_mt, std_mt, sm, champ, mean_fiscal, fiscal_frac, pyrr, delta))
            else:
                verdict = 'PASS-PYRRHIC'
                pass_pyrr.append((cand_tag, mult, opt_eta, mean_mt, std_mt, sm, champ, mean_fiscal, fiscal_frac, pyrr, delta))
        else:
            failed = []
            if not c1: failed.append('hub')
            if not c2: failed.append('chmp')
            if not c3: failed.append('s/m')
            if not c4: failed.append('delta')
            verdict = f'FAIL({"+".join(failed)})'
            fail_cells.append((cand_tag, mult, opt_eta, mean_mt, std_mt, sm, champ, mean_fiscal, fiscal_frac, pyrr, delta))

        print(f'{cand_tag:>5} | {mult:>5.1f} | {opt_eta:>5.2f} | {mts_str:>30} | '
              f'{mean_mt:>6.1f} | {std_mt:>5.1f} | {sm:>5.2f} | {champ:>4} | '
              f'{mean_fiscal:>5.0f} | {fiscal_frac:>5.2f} | {"T" if pyrr else "F":>4} | '
              f'{delta:>+7.0f} | {verdict}', flush=True)

    print('', flush=True)
    print('=== Summary by group (ranked by deepest delta within group) ===', flush=True)
    pass_clean.sort(key=lambda w: w[10])
    pass_pyrr.sort(key=lambda w: w[10])
    fail_cells.sort(key=lambda w: w[10])

    if pass_clean:
        print(f'PASS-CLEAN ({len(pass_clean)} cells):', flush=True)
        for w in pass_clean:
            print(f'  cand={w[0]} mult={w[1]} eta*={w[2]:.2f} | max_ten={w[3]:.1f}+/-{w[4]:.1f} (s/m={w[5]:.2f}) | '
                  f'fiscal={w[7]:.0f} ({w[8]:.0%}) | delta={w[10]:+.0f}', flush=True)
    else:
        print('PASS-CLEAN: (none)', flush=True)

    if pass_pyrr:
        print(f'PASS-PYRRHIC ({len(pass_pyrr)} cells):', flush=True)
        for w in pass_pyrr:
            print(f'  cand={w[0]} mult={w[1]} eta*={w[2]:.2f} | max_ten={w[3]:.1f}+/-{w[4]:.1f} (s/m={w[5]:.2f}) | '
                  f'fiscal={w[7]:.0f} ({w[8]:.0%}) | delta={w[10]:+.0f}', flush=True)
    else:
        print('PASS-PYRRHIC: (none)', flush=True)

    print(f'FAIL: {len(fail_cells)} cells (suppressed for brevity; see per-cell table above)', flush=True)

    print('', flush=True)
    if pass_clean:
        w = pass_clean[0]
        print(f'OUTCOME A (clean winner): cand={w[0]} mult={w[1]} eta*={w[2]:.2f}', flush=True)
        print(f'  -> next: 15-seed validation at this cell (Step 8)', flush=True)
    elif pass_pyrr:
        w = pass_pyrr[0]
        print(f'OUTCOME B (Pyrrhic at all passing cells): cand={w[0]} mult={w[1]} eta*={w[2]:.2f}', flush=True)
        print(f'  -> next: fall back to w58 cv=0 with cost-amplification chapter section', flush=True)
    else:
        # Distinguish C from D: clean hub anywhere?
        clean_hubs = [c for c in fail_cells if 15 <= c[3] <= 50 and c[5] < 0.35 and c[6] == 0]
        if clean_hubs:
            print(f'OUTCOME C (clean hub but claim 3 dead): {len(clean_hubs)} cells with clean hub but delta>=0', flush=True)
            print(f'  -> next: fall back to w58 cv=0 with hub-vs-saving foreclosure', flush=True)
        else:
            print(f'OUTCOME D (bimodal/lock-in at all cells): rich-gets-richer independent of init scheme', flush=True)
            print(f'  -> next: fall back to w58 cv=0 with strongest foreclosure framing', flush=True)


if __name__ == '__main__':
    main()
