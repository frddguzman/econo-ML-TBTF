"""Brini-style hub diagnostics for D1 dream-zone cells.

Single PNG with rows = cells, columns = seeds (full t=0..1000 per panel).
Goal: visually verify whether these are truly multi-hub rotation or
champion-then-collapse like the truncation experiments.

Cells (all D1 = hetero + reintroduce_with_median=False, no truncation):
  1. bsl cv=1.0 eta=0.00  (max=164, avg=5.3, s/m=0.39, DREAM)
  2. bsl cv=1.0 eta=0.85  (max=174, avg=6.4, s/m=0.41, DREAM)
  3. w58 cv=0.7 eta=0.85  (max=109, avg=6.2, s/m=0.44, lowest max_ten DREAM)
  4. w58 cv=1.0 eta=0.10  (max=165, avg=10.2, s/m=0.55, highest avg_ten)
  5. w58 cv=1.0 eta=0.85  (max=148, avg=10.0, s/m=0.50, dream + high avg)

Total: 5 cells x 5 seeds = 25 sims, ~30 sec.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]
N = 50
WORKERS = 6
PROJ = os.path.dirname(os.path.abspath(__file__))

# (label, mu, omega, gamma, cv, eta)
CELLS = [
    ('bsl cv=1.0 eta=0.00 (DREAM low-eta)', 0.70, 0.50, 0.10, 1.0, 0.00),
    ('bsl cv=1.0 eta=0.85 (DREAM high-eta)', 0.70, 0.50, 0.10, 1.0, 0.85),
    ('w58 cv=0.7 eta=0.85 (DREAM lowest-max)', 0.70, 0.58, 0.10, 0.7, 0.85),
    ('w58 cv=1.0 eta=0.10 (DREAM highest-avg)', 0.70, 0.58, 0.10, 1.0, 0.10),
    ('w58 cv=1.0 eta=0.85 (DREAM)', 0.70, 0.58, 0.10, 1.0, 0.85),
]


def run_one(args):
    mu, omega, gamma, cv, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = cv
    cfg['reintroduce_with_median'] = False
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    hub_id  = np.array([float(v) if v is not None and v >= 0 else np.nan for v in s.best_lender[:T]])
    hub_fit = np.array([float(v) if v is not None and v >= 0 else np.nan for v in s.best_lender_fitness[:T]])
    hub_cli = np.array([float(v) if v is not None and v >= 0 else np.nan for v in s.best_lender_clients[:T]])
    total_bk = int(np.nansum(s.bankruptcy[:T]))
    return {
        'cv': cv, 'eta': eta, 'omega': omega, 'seed': seed, 'T': T,
        'hub_id': hub_id, 'hub_fit': hub_fit, 'hub_cli': hub_cli,
        'total_bk': total_bk,
    }


def compute_tenure_stats(hub_id_arr):
    raw = [int(v) for v in hub_id_arr if not np.isnan(v) and v >= 0]
    runs = []
    if raw:
        prev = raw[0]; rl = 1
        for k in raw[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    avg_ten = sum(runs)/len(runs) if runs else 0
    turnovers = max(0, len(runs) - 1)
    return max_ten, avg_ten, turnovers


def main():
    jobs = [(mu, omega, gamma, cv, eta, s) for (_, mu, omega, gamma, cv, eta) in CELLS for s in SEEDS]
    n_jobs = len(jobs)
    print(f'Running {n_jobs} sims (5 cells x 5 seeds)...', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            results.append(r)
            if i % 5 == 0:
                print(f'  ... {i}/{n_jobs} done', flush=True)

    print('Building compact figure...', flush=True)
    n_rows = len(CELLS)
    n_cols = len(SEEDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.6 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1: axes = axes.reshape(1, n_cols)
    if n_cols == 1: axes = axes.reshape(n_rows, 1)

    for row, (label, mu, omega, gamma, cv, eta) in enumerate(CELLS):
        for col, seed in enumerate(SEEDS):
            r = next((rr for rr in results if abs(rr['cv']-cv)<1e-6
                     and abs(rr['eta']-eta)<1e-9 and abs(rr['omega']-omega)<1e-6
                     and rr['seed']==seed), None)
            ax = axes[row, col]
            if r is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue
            T = r['T']
            cli_max = max(20.0, float(np.nanmax(r['hub_cli'])) if not np.all(np.isnan(r['hub_cli'])) else 20.0)
            hub_id_n  = r['hub_id'] / N
            hub_cli_n = r['hub_cli'] / cli_max
            max_ten, avg_ten, turnovers = compute_tenure_stats(r['hub_id'])
            times = np.arange(T)
            ax.plot(times, hub_id_n,  color='black', linewidth=0.8, alpha=0.9)
            ax.plot(times, hub_cli_n, color='red',   linestyle='--', linewidth=0.9, alpha=0.7)
            ax.plot(times, r['hub_fit'], color='green', linestyle=':', linewidth=0.8, alpha=0.6)
            ax.set_xlim(0, T); ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25, linewidth=0.4)
            ax.set_title(f'seed={seed}\nmax={max_ten} avg={avg_ten:.1f} turn={turnovers} bk={r["total_bk"]}',
                         fontsize=8.5, loc='left')
            if row == n_rows - 1: ax.set_xlabel('Time', fontsize=8)
            if col == 0: ax.set_ylabel(f'{label}\n\nhub-id (norm) / clients / fit', fontsize=7.5)
            ax.tick_params(labelsize=7)

    handles = [plt.Line2D([0],[0], color='black', linewidth=1.2, label='Hub ID (norm /N)'),
               plt.Line2D([0],[0], color='red', linestyle='--', linewidth=1.2, label='Clients (norm /max)'),
               plt.Line2D([0],[0], color='green', linestyle=':', linewidth=1.2, label='Hub fitness')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.005), fontsize=9)
    fig.suptitle('D1 dream-zone cells: hetero + reintroduce_with_median=False (no truncation)\n'
                 'Rows = (cand, cv, eta) · Columns = seed · single figure for visual comparison',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(PROJ, 'd1_dream_zone_brini.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {out_path}', flush=True)


if __name__ == '__main__':
    main()
