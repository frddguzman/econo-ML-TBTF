"""Reproduce Brini Fig. 6 layout at w58 + cv=1.0 + eta=0.2 for two seeds:
  - 26464: champion seed (max_ten=472, avg_ten=56, turnovers=17)
  - 26463: churn seed   (max_ten=62,  avg_ten=5,  turnovers=198)

Layout matches Brini's: 1 row, 2 panels (left t=0..500, right t=500..1000).
Three series overlaid:
  - black solid line: hub bank ID (normalized to [0,1] by /N)
  - red dashed line: hub client count (in-degree, normalized to [0,1] by /max_observed)
  - green dotted line: hub fitness (already in [0,1])
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

MU = 0.70
OMEGA = 0.58
GAMMA = 0.10
CV = 1.0
ETA = 0.2
REGIME = 'socialized_tax'
SEEDS = [26463, 26464]
N = 50


def run_one(seed):
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=ETA, regime=REGIME)
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = CV
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
    return seed, T, hub_id, hub_fit, hub_cli


def plot_one(seed, T, hub_id, hub_fit, hub_cli, label):
    # Compute statistics for the title panel
    raw_ids = [int(v) for v in hub_id if not np.isnan(v) and v >= 0]
    runs = []
    if raw_ids:
        prev = raw_ids[0]; rl = 1
        for k in raw_ids[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    avg_ten = sum(runs)/len(runs) if runs else 0
    turnovers = max(0, len(runs) - 1)
    avg_cli = float(np.nanmean(hub_cli))
    cli_max = max(20.0, float(np.nanmax(hub_cli)) if not np.all(np.isnan(hub_cli)) else 20.0)

    # Normalize
    hub_id_n  = hub_id / N
    hub_cli_n = hub_cli / cli_max

    # 2-panel split: t=0..500, t=500..1000
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    times = np.arange(T)
    cuts  = [(0, 500), (500, 1000)]
    for ax, (lo, hi) in zip(axes, cuts):
        sl = slice(lo, hi)
        ax.plot(times[sl], hub_id_n[sl],  color='black', linestyle='-',  linewidth=1.0, label='Hub ID (norm)')
        ax.plot(times[sl], hub_cli_n[sl], color='red',   linestyle='--', linewidth=1.2, label='In-degree (norm)')
        ax.plot(times[sl], hub_fit[sl],   color='green', linestyle=':',  linewidth=1.0, label='Hub fitness')
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Time')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Hub ID, in-degree, fit')
    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.85)

    fig.suptitle(
        f'w58 + cv=1.0 + eta=0.2 + seed={seed} ({label})  |  '
        f'max_ten={max_ten}  avg_ten={avg_ten:.1f}  turnovers={turnovers}  avg_clients={avg_cli:.1f}',
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'brini_w58_h1_e02_seed{seed}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)

    # Save the 3 series too in case you want to re-plot
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'brini_w58_h1_e02_seed{seed}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t', 'hub_id', 'hub_id_norm', 'hub_clients', 'hub_clients_norm', 'hub_fitness'])
        for t in range(T):
            w.writerow([t, hub_id[t], hub_id_n[t], hub_cli[t], hub_cli_n[t], hub_fit[t]])
    print(f'  seed={seed} ({label}): max_ten={max_ten} avg_ten={avg_ten:.1f} turnovers={turnovers} avg_cli={avg_cli:.1f}  ->  {out}')


def main():
    print(f'Brini Fig.6 reproduction: w58 + cv={CV} + eta={ETA} + social, seeds {SEEDS}')
    labels = {26463: 'churn', 26464: 'champion'}
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run_one, SEEDS))
    for seed, T, hub_id, hub_fit, hub_cli in results:
        plot_one(seed, T, hub_id, hub_fit, hub_cli, labels.get(seed, ''))
    print('Done.')


if __name__ == '__main__':
    main()
