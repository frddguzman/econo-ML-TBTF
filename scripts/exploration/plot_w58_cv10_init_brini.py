"""Brini-style hub diagnostics for w58 cv=1.0 + reintroduce_with_median=False
at the user-flagged etas: 0.5 (transition) and 0.9 (cleanest DREAM, s/m=0.23).

Single PNG, rows = etas, cols = seeds. Shows full t=0..1000 per panel.
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
ETAS = [0.5, 0.9]
N = 50
WORKERS = 6
PROJ = os.path.dirname(os.path.abspath(__file__))


def run_one(args):
    eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = 1.0
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
        'eta': eta, 'seed': seed, 'T': T,
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
    jobs = [(e, s) for e in ETAS for s in SEEDS]
    n_jobs = len(jobs)
    print(f'Running {n_jobs} sims (2 etas x 5 seeds)...', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            results.append(r)
            if i % 5 == 0:
                print(f'  ... {i}/{n_jobs} done', flush=True)

    print('Building figure...', flush=True)
    n_rows = len(ETAS)
    n_cols = len(SEEDS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.0 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1: axes = axes.reshape(1, n_cols)

    for row, eta in enumerate(ETAS):
        for col, seed in enumerate(SEEDS):
            r = next((rr for rr in results if abs(rr['eta']-eta)<1e-9 and rr['seed']==seed), None)
            ax = axes[row, col]
            if r is None: continue
            T = r['T']
            cli_max = max(20.0, float(np.nanmax(r['hub_cli'])) if not np.all(np.isnan(r['hub_cli'])) else 20.0)
            hub_id_n  = r['hub_id'] / N
            hub_cli_n = r['hub_cli'] / cli_max
            max_ten, avg_ten, turnovers = compute_tenure_stats(r['hub_id'])
            times = np.arange(T)
            ax.plot(times, hub_id_n,  color='black', linewidth=0.9, alpha=0.95)
            ax.plot(times, hub_cli_n, color='red',   linestyle='--', linewidth=1.0, alpha=0.7)
            ax.plot(times, r['hub_fit'], color='green', linestyle=':', linewidth=0.9, alpha=0.7)
            ax.set_xlim(0, T); ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25, linewidth=0.4)
            ax.set_title(f'seed={seed} | max={max_ten} avg={avg_ten:.1f} turn={turnovers} bk={r["total_bk"]}',
                         fontsize=9, loc='left')
            if row == n_rows - 1: ax.set_xlabel('Time', fontsize=8)
            if col == 0:
                regime_label = 'η=0.50 (transition)' if eta == 0.5 else 'η=0.90 (cleanest DREAM, s/m=0.23)'
                ax.set_ylabel(f'{regime_label}\n\nhub-id (norm) / clients / fit', fontsize=9)
            ax.tick_params(labelsize=8)

    handles = [plt.Line2D([0],[0], color='black', linewidth=1.5, label='Hub ID (norm /N)'),
               plt.Line2D([0],[0], color='red', linestyle='--', linewidth=1.5, label='Clients (norm /max)'),
               plt.Line2D([0],[0], color='green', linestyle=':', linewidth=1.5, label='Hub fitness')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.005), fontsize=10)
    fig.suptitle('w58 cv=1.0 + reintroduce_with_median=False at eta=0.5 and eta=0.9 (5 seeds each)\n'
                 'Looking for visible champion phases vs multi-hub rotation',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(PROJ, 'w58_cv10_init_brini_e05_e09.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {out_path}', flush=True)


if __name__ == '__main__':
    main()
