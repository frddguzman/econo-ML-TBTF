"""Brini-style hub diagnostics for w58 cv=1.0 + reintroduce_with_median=False
across ALL 15 etas at the canonical dashboard seed (26474).

Layout: 3 rows x 5 cols = 15 panels, each showing t=0..1000.
Allows scanning eta progression at a single consistent seed.
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

SEED = 26474
ETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
N = 50
WORKERS = 6
PROJ = os.path.dirname(os.path.abspath(__file__))


def run_one(eta):
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
    m.initialize(seed=SEED, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    hub_id = np.array([float(v) if v is not None and v >= 0 else np.nan for v in s.best_lender[:T]])
    hub_fit = np.array([float(v) if v is not None and v >= 0 else np.nan for v in s.best_lender_fitness[:T]])
    hub_cli = np.array([float(v) if v is not None and v >= 0 else np.nan for v in s.best_lender_clients[:T]])
    total_bk = int(np.nansum(s.bankruptcy[:T]))
    return {'eta': eta, 'T': T, 'hub_id': hub_id, 'hub_fit': hub_fit, 'hub_cli': hub_cli, 'total_bk': total_bk}


def stats(hub_id_arr):
    raw = [int(v) for v in hub_id_arr if not np.isnan(v) and v >= 0]
    runs = []
    if raw:
        prev = raw[0]; rl = 1
        for k in raw[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    return (max(runs) if runs else 0,
            sum(runs)/len(runs) if runs else 0,
            max(0, len(runs) - 1))


def main():
    print(f'Running {len(ETAS)} sims at seed={SEED}...', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run_one, ETAS))
    print('Building figure...', flush=True)

    fig, axes = plt.subplots(3, 5, figsize=(22, 11), sharex=True, sharey=True)
    for idx, r in enumerate(results):
        row, col = idx // 5, idx % 5
        ax = axes[row, col]
        T = r['T']
        cli_max = max(20.0, float(np.nanmax(r['hub_cli'])) if not np.all(np.isnan(r['hub_cli'])) else 20.0)
        hub_id_n = r['hub_id'] / N
        hub_cli_n = r['hub_cli'] / cli_max
        max_ten, avg_ten, turn = stats(r['hub_id'])
        times = np.arange(T)
        ax.plot(times, hub_id_n, color='black', linewidth=0.9, alpha=0.95)
        ax.plot(times, hub_cli_n, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.plot(times, r['hub_fit'], color='green', linestyle=':', linewidth=0.9, alpha=0.7)
        ax.set_xlim(0, T); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25, linewidth=0.4)
        title = f"eta={r['eta']:.2f} | max={max_ten} avg={avg_ten:.1f} turn={turn} bk={r['total_bk']}"
        ax.set_title(title, fontsize=10, loc='left')
        if row == 2: ax.set_xlabel('Time', fontsize=9)
        if col == 0: ax.set_ylabel('hub-id (norm) / clients / fit', fontsize=9)
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([0],[0], color='black', linewidth=1.5, label='Hub ID (norm /N)'),
               plt.Line2D([0],[0], color='red', linestyle='--', linewidth=1.5, label='Clients (norm /max)'),
               plt.Line2D([0],[0], color='green', linestyle=':', linewidth=1.5, label='Hub fitness')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.005), fontsize=11)
    fig.suptitle(f'w58 cv=1.0 + reintroduce_with_median=False at SEED={SEED} (dashboard seed) — all 15 etas',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(PROJ, f'w58_d1_seed{SEED}_all_eta.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {out_path}', flush=True)


if __name__ == '__main__':
    main()
