"""Brini-style at the MOST bimodal D1 cell: bsl cv=1.0 init_replace eta=0.70.

Per-seed max_ten: [267, 820, 230, 124, 117]
- seed 26463: max=820 (champion dominates 82% of run)
- seed 26462: max=267 (smaller star)
- seeds 26464, 26465, 26466: mid-tenure rotation

5 panels in one row = full width per panel, easy to read.
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
WORKERS = 5
PROJ = os.path.dirname(os.path.abspath(__file__))


def run_one(seed):
    cfg = ddr.make_config(basis='equity', omega=0.50, eta=0.70, regime='socialized_tax')
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
    return {'seed': seed, 'T': T, 'hub_id': hub_id, 'hub_fit': hub_fit, 'hub_cli': hub_cli, 'total_bk': total_bk}


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
    print(f'Running 5 sims at bsl cv=1.0 init_replace eta=0.70...', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run_one, SEEDS))
    print('Done. Building figure...', flush=True)

    fig, axes = plt.subplots(5, 1, figsize=(15, 18), sharex=True, sharey=True)
    for row, r in enumerate(sorted(results, key=lambda x: x['seed'])):
        ax = axes[row]
        T = r['T']
        cli_max = max(20.0, float(np.nanmax(r['hub_cli'])) if not np.all(np.isnan(r['hub_cli'])) else 20.0)
        hub_id_n  = r['hub_id'] / N
        hub_cli_n = r['hub_cli'] / cli_max
        max_ten, avg_ten, turn = stats(r['hub_id'])
        times = np.arange(T)
        ax.plot(times, hub_id_n,  color='black', linewidth=1.0, alpha=0.95, label='Hub ID (norm /N)' if row==0 else None)
        ax.plot(times, hub_cli_n, color='red',   linestyle='--', linewidth=1.1, alpha=0.7, label='Clients (norm /max)' if row==0 else None)
        ax.plot(times, r['hub_fit'], color='green', linestyle=':', linewidth=1.0, alpha=0.7, label='Hub fitness' if row==0 else None)
        ax.set_xlim(0, T); ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_title(f"seed={r['seed']}  |  max_ten={max_ten}  avg_ten={avg_ten:.1f}  turnovers={turn}  total_bk={r['total_bk']}",
                     fontsize=11, loc='left')
        ax.set_ylabel('hub-id (norm) / clients / fit', fontsize=10)
        if row == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    axes[-1].set_xlabel('Time', fontsize=11)
    fig.suptitle('Most bimodal D1 cell: bsl cv=1.0 + reintroduce_with_median=False, eta=0.70 (5 seeds)\n'
                 'Per-seed max_ten = [267, 820, 230, 124, 117] — seed 26463 has 820-period champion',
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out_path = os.path.join(PROJ, 'd1_bimodal_bsl_cv10_e070.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {out_path}', flush=True)


if __name__ == '__main__':
    main()
