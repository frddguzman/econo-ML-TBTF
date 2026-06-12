"""V3 comparison: same as v2 but using SEED=26465 (median seed for both C3 cv=0.7 cells).

Per per-seed inspection at the user's picks:
  - C3 g=0.05 cv=0.7 eta=0.3: max_tens [12, 18, 164, 378, 641] -> median = 164 -> SEED=26465
  - C3 g=0.12 cv=0.7 eta=0.2: max_tens [9, 10, 16, 687, 916]   -> median = 16  -> SEED=26465

So SEED=26465 is the median for BOTH cells, just by coincidence. At this seed:
  - g=0.05 cv=0.7 is in "partial centralization" mode (max_ten 164)
  - g=0.12 cv=0.7 is in churn mode (max_ten 16)
This is the honest representation of typical model behavior at these cells.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

# Project root (the repo directory itself).
PROJ = Path(__file__).resolve().parent

# Output / simulation data directory. Override via TBTF_SIM_DIR env var.
# Default: a sibling 'Simulations/' folder next to the repo root.
SIM = Path(os.environ.get(
    'TBTF_SIM_DIR',
    str(Path(__file__).resolve().parent.parent / 'Simulations'),
))
N = 50

MEDIAN_SEED = 26465
W58_SEED = 26474


def run_one(args):
    label, mu, omega, gamma, cv, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    cfg['equity_heterogeneity'] = (cv > 0)
    cfg['equity_cv'] = cv
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
    equity  = np.array([float(v) if v is not None else np.nan for v in s.equity[:T]])
    return label, T, hub_id, hub_fit, hub_cli, equity


def load_w58_dash(eta_tag):
    path = os.path.join(SIM, f'dash_w58_st_{eta_tag}.csv')
    hub_id, hub_fit, hub_cli, equity = [], [], [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            try: hub_id.append(float(r['best_lender']) if r['best_lender'] not in ('', 'nan', 'None') else np.nan)
            except: hub_id.append(np.nan)
            try: hub_fit.append(float(r['best_lender_fitness']))
            except: hub_fit.append(np.nan)
            try: hub_cli.append(float(r['best_lender_clients']))
            except: hub_cli.append(np.nan)
            try: equity.append(float(r['equity']))
            except: equity.append(np.nan)
    return len(hub_id), np.array(hub_id), np.array(hub_fit), np.array(hub_cli), np.array(equity)


def plot_brini(results, out_path):
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3.8 * len(results)), sharey='row')
    if len(results) == 1: axes = axes.reshape(1, 2)
    for row, (label, T, hub_id, hub_fit, hub_cli) in enumerate(results):
        cli_max = max(20.0, float(np.nanmax(hub_cli)) if not np.all(np.isnan(hub_cli)) else 20.0)
        hub_id_n  = hub_id / N
        hub_cli_n = hub_cli / cli_max
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
        times = np.arange(T)
        for col, (lo, hi) in enumerate([(0, 500), (500, 1000)]):
            ax = axes[row, col]
            sl = slice(lo, hi)
            ax.plot(times[sl], hub_id_n[sl],  color='black', linewidth=1.0, label='Hub ID (norm)' if col == 0 else None)
            ax.plot(times[sl], hub_cli_n[sl], color='red', linestyle='--', linewidth=1.2, label='In-degree (norm)' if col == 0 else None)
            ax.plot(times[sl], hub_fit[sl],   color='green', linestyle=':', linewidth=1.0, label='Hub fitness' if col == 0 else None)
            ax.set_xlim(lo, hi); ax.set_ylim(0, 1.05)
            if row == len(results) - 1: ax.set_xlabel('Time')
            if col == 0: ax.set_ylabel('Hub ID, in-degree, fit')
            ax.grid(alpha=0.3)
        axes[row, 0].legend(loc='upper right', fontsize=8, framealpha=0.85)
        title_text = f'{label}  |  max_ten={max_ten}  avg_ten={avg_ten:.1f}  turnovers={turnovers}  avg_cli={avg_cli:.1f}'
        axes[row, 0].set_title(title_text, fontsize=10, loc='left')
    fig.suptitle('Brini Fig.6 reproduction — MEDIAN seed (representative model behavior)', fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def plot_equity(results_with_equity, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ['#1a3f6b', '#27ae60', '#c0392b']
    for (label, T, equity), color in zip(results_with_equity, colors):
        ax.plot(np.arange(T), equity, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('System equity Σ E_i', fontsize=12)
    ax.set_title('System equity over time — MEDIAN seed (representative model behavior)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def main():
    print(f'Running 2 sims at MEDIAN seed (SEED={MEDIAN_SEED}) for the C3 cv=0.7 cells ...')
    jobs = [
        (f'C3 γ=0.05 cv=0.7 η=0.3 (seed={MEDIAN_SEED} median)', 0.6, 0.7, 0.05, 0.7, 0.3, MEDIAN_SEED),
        (f'C3 γ=0.12 cv=0.7 η=0.2 (seed={MEDIAN_SEED} median)', 0.6, 0.7, 0.12, 0.7, 0.2, MEDIAN_SEED),
    ]
    with ProcessPoolExecutor(max_workers=2) as ex:
        c3_results = list(ex.map(run_one, jobs))

    print('Loading w58 cv=0 from existing dash CSV ...')
    w58_T, w58_id, w58_fit, w58_cli, w58_eq = load_w58_dash('e01')

    fig_brini = [(f'w58 cv=0 η=0.1 (SEED={W58_SEED})', w58_T, w58_id, w58_fit, w58_cli)]
    fig_eq    = [(f'w58 cv=0 η=0.1 (SEED={W58_SEED})', w58_T, w58_eq)]
    for label, T, hub_id, hub_fit, hub_cli, equity in c3_results:
        fig_brini.append((label, T, hub_id, hub_fit, hub_cli))
        fig_eq.append((label, T, equity))

    plot_brini(fig_brini, os.path.join(PROJ, 'compare_v3_fig2_brini_median.png'))
    plot_equity(fig_eq,   os.path.join(PROJ, 'compare_v3_fig3_equity_median.png'))
    print('\nDone.')


if __name__ == '__main__':
    main()
