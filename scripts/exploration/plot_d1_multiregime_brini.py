"""Brini-style hub diagnostics for w58 cv=1.0 + reintroduce_with_median=False
across no_tax + 3 resolution_fund tau variants at SEED=26474.

Layout: 4 rows (regimes) x 10 cols (etas 0.0 to 0.9, 0.1 step).
Each panel shows full t=0..1000 with hub_id, clients, fitness.
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
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N = 50
WORKERS = 6
PROJ = os.path.dirname(os.path.abspath(__file__))

# (label, regime, tau_fund_override)
REGIME_VARIANTS = [
    ('no_tax',                   'none',            None),
    ('rf tau=1e-4 (default)',    'resolution_fund', 1e-4),
    ('rf tau=1e-5 (canonical)',  'resolution_fund', 1e-5),
    ('rf tau=1e-6',              'resolution_fund', 1e-6),
]


def run_one(args):
    label, regime, tau, eta = args
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=eta, regime=regime)
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = 1.0
    cfg['reintroduce_with_median'] = False
    if tau is not None:
        cfg['fund_levy_rate'] = tau
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
    return {
        'label': label, 'regime': regime, 'tau': tau, 'eta': eta, 'T': T,
        'hub_id': hub_id, 'hub_fit': hub_fit, 'hub_cli': hub_cli,
        'total_bk': total_bk,
    }


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
    jobs = [(label, regime, tau, eta)
            for (label, regime, tau) in REGIME_VARIANTS
            for eta in ETAS]
    n_jobs = len(jobs)
    print(f'Running {n_jobs} sims (4 regimes x 10 etas) at seed={SEED}...', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(run_one, jobs))
    print('Building figure...', flush=True)

    n_rows = len(REGIME_VARIANTS)
    n_cols = len(ETAS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 2.5 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1: axes = axes.reshape(1, n_cols)
    if n_cols == 1: axes = axes.reshape(n_rows, 1)

    for row, (label, regime, tau) in enumerate(REGIME_VARIANTS):
        for col, eta in enumerate(ETAS):
            r = next((rr for rr in results if rr['label']==label and abs(rr['eta']-eta)<1e-9), None)
            ax = axes[row, col]
            if r is None: continue
            T = r['T']
            cli_max = max(20.0, float(np.nanmax(r['hub_cli'])) if not np.all(np.isnan(r['hub_cli'])) else 20.0)
            hub_id_n = r['hub_id'] / N
            hub_cli_n = r['hub_cli'] / cli_max
            max_ten, avg_ten, turn = stats(r['hub_id'])
            times = np.arange(T)
            ax.plot(times, hub_id_n, color='black', linewidth=0.7, alpha=0.95)
            ax.plot(times, hub_cli_n, color='red', linestyle='--', linewidth=0.7, alpha=0.7)
            ax.plot(times, r['hub_fit'], color='green', linestyle=':', linewidth=0.7, alpha=0.7)
            ax.set_xlim(0, T); ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25, linewidth=0.4)
            ax.set_title(f"η={eta:.1f} max={max_ten} avg={avg_ten:.1f}\nturn={turn} bk={r['total_bk']}",
                         fontsize=8, loc='left')
            if row == n_rows - 1: ax.set_xlabel('Time', fontsize=8)
            if col == 0: ax.set_ylabel(label, fontsize=10, fontweight='bold')
            ax.tick_params(labelsize=7)

    handles = [plt.Line2D([0],[0], color='black', linewidth=1.5, label='Hub ID (norm /N)'),
               plt.Line2D([0],[0], color='red', linestyle='--', linewidth=1.5, label='Clients (norm /max)'),
               plt.Line2D([0],[0], color='green', linestyle=':', linewidth=1.5, label='Hub fitness')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.005), fontsize=10)
    fig.suptitle(f'w58 cv=1.0 + reintroduce_with_median=False at SEED={SEED}\n'
                 f'Rows = fiscal regime/τ · Columns = η',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(PROJ, f'd1_multiregime_brini_seed{SEED}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {out_path}', flush=True)


if __name__ == '__main__':
    main()
