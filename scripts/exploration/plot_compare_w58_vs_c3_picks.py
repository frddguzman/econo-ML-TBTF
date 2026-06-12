"""Generate 3 comparison figures: w58 cv=0 vs C3 + g=0.05/0.12 + cv=0.70.

Figure 1: Social eta-sweep (multi-seed mean from existing CSVs)
Figure 2: Brini Fig.6 hub overlay (single-seed at SEED=26474, eta=eta*)
Figure 3: System equity time-series (same single-seed, same eta*)

Cells:
  - w58 cv=0 at eta=0.1 social (existing dash_w58_st_e01.csv)
  - c3 g=0.05 cv=0.70 at eta=0.3 social (NEW sim required)
  - c3 g=0.12 cv=0.70 at eta=0.2 social (NEW sim required)

For multi-seed mean of the C3+cv=0.7 cells we read from
sweep_c3_hetero_full_eta_raw.csv. For w58 we read thesis_lehman_w58.csv.
For Brini-style + equity time-series at the C3 cells, we run 2 fresh single-seed
sims at SEED=26474 and capture per-period data.
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
PROJ = str(Path(__file__).resolve().parent)

# Output / simulation data directory. Override via TBTF_SIM_DIR env var.
# Default: a sibling 'Simulations/' folder next to the repo root.
SIM = os.environ.get(
    'TBTF_SIM_DIR',
    str(Path(__file__).resolve().parent.parent / 'Simulations'),
)
SEED = 26474   # dashboard convention single-seed
N = 50

# ============================================================================
# Figure 1 — Social eta-sweep (multi-seed mean)
# ============================================================================

def load_w58_eta_sweep(path):
    """thesis_lehman_w58.csv has cv=0 implicit. Pulls 5-seed mean per eta in social."""
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r['fiscal_regime'] == 'socialized_tax':
                rows.append(r)
    etas = sorted(set(float(r['eta']) for r in rows))
    means, stds = [], []
    for eta in etas:
        bks = [int(r['total_bk']) for r in rows if abs(float(r['eta']) - eta) < 1e-6]
        m = sum(bks) / len(bks)
        s = (sum((x - m) ** 2 for x in bks) / max(1, len(bks) - 1)) ** 0.5 if len(bks) > 1 else 0
        means.append(m); stds.append(s)
    return etas, means, stds


def load_c3_eta_sweep(path, gamma, cv):
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            if (abs(float(r['gamma']) - gamma) < 1e-6
                and abs(float(r['cv']) - cv) < 1e-6
                and r['fiscal_regime'] == 'socialized_tax'):
                rows.append(r)
    etas = sorted(set(float(r['eta']) for r in rows))
    means, stds = [], []
    for eta in etas:
        bks = [int(r['total_bk']) for r in rows if abs(float(r['eta']) - eta) < 1e-6]
        m = sum(bks) / len(bks)
        s = (sum((x - m) ** 2 for x in bks) / max(1, len(bks) - 1)) ** 0.5 if len(bks) > 1 else 0
        means.append(m); stds.append(s)
    return etas, means, stds


def plot_fig1_eta_sweep():
    w58_etas, w58_means, w58_stds = load_w58_eta_sweep(os.path.join(SIM, 'thesis_lehman_w58.csv'))
    c3_path = os.path.join(PROJ, 'sweep_c3_hetero_full_eta_raw.csv')
    g05_etas, g05_means, g05_stds = load_c3_eta_sweep(c3_path, 0.05, 0.7)
    g12_etas, g12_means, g12_stds = load_c3_eta_sweep(c3_path, 0.12, 0.7)

    fig, ax = plt.subplots(figsize=(11, 6))

    # w58
    ax.errorbar(w58_etas, w58_means, yerr=w58_stds, label='w58 cv=0 (μ=0.7,ω=0.58)',
                color='#1a3f6b', marker='s', markersize=6, linewidth=2, capsize=4)
    # c3 g=0.05 cv=0.7
    ax.errorbar(g05_etas, g05_means, yerr=g05_stds, label='C3 γ=0.05 cv=0.7 (μ=0.6,ω=0.7)',
                color='#27ae60', marker='o', markersize=6, linewidth=2, capsize=4)
    # c3 g=0.12 cv=0.7
    ax.errorbar(g12_etas, g12_means, yerr=g12_stds, label='C3 γ=0.12 cv=0.7 (μ=0.6,ω=0.7)',
                color='#c0392b', marker='^', markersize=6, linewidth=2, capsize=4)

    # Mark optima
    for etas, means, color, label_eta_star in [
        (w58_etas, w58_means, '#1a3f6b', 0.1),
        (g05_etas, g05_means, '#27ae60', 0.3),
        (g12_etas, g12_means, '#c0392b', 0.2),
    ]:
        idx = etas.index(label_eta_star)
        ax.scatter([label_eta_star], [means[idx]], s=200, facecolors='none',
                   edgecolors=color, linewidths=2, zorder=5)

    ax.set_xlabel('η (bailout coverage)', fontsize=12)
    ax.set_ylabel('Total bankruptcies (5-seed mean ± std)', fontsize=12)
    ax.set_title('Social regime η-sweep: w58 cv=0 vs C3 γ cv=0.7 picks', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(PROJ, 'compare_fig1_social_eta_sweep.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


# ============================================================================
# Figures 2, 3 — single-seed time-series (Brini-style + equity)
# ============================================================================

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


def load_w58_dash():
    """Read existing dash_w58_st_e01.csv for w58 cv=0 at eta=0.1 social, seed 26474."""
    path = os.path.join(SIM, 'dash_w58_st_e01.csv')
    hub_id, hub_fit, hub_cli, equity = [], [], [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                hub_id.append(float(r['best_lender']) if r['best_lender'] not in ('', 'nan', 'None') else np.nan)
            except (ValueError, KeyError):
                hub_id.append(np.nan)
            try:
                hub_fit.append(float(r['best_lender_fitness']))
            except (ValueError, KeyError, TypeError):
                hub_fit.append(np.nan)
            try:
                hub_cli.append(float(r['best_lender_clients']))
            except (ValueError, KeyError, TypeError):
                hub_cli.append(np.nan)
            try:
                equity.append(float(r['equity']))
            except (ValueError, KeyError, TypeError):
                equity.append(np.nan)
    return ('w58 cv=0 η=0.1 (seed=26474)',
            len(hub_id), np.array(hub_id), np.array(hub_fit),
            np.array(hub_cli), np.array(equity))


def plot_fig2_brini(results):
    """3 stacked panels, one per cell. Each panel: 2 sub-panels (left t=0..500, right t=500..1000).
    Each panel shows hub_id (norm), hub_clients (norm), hub_fitness as in Brini Fig.6.
    """
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3.8 * len(results)),
                             sharey='row')
    if len(results) == 1:
        axes = axes.reshape(1, 2)

    for row, (label, T, hub_id, hub_fit, hub_cli) in enumerate(results):
        cli_max = max(20.0, float(np.nanmax(hub_cli)) if not np.all(np.isnan(hub_cli)) else 20.0)
        hub_id_n  = hub_id / N
        hub_cli_n = hub_cli / cli_max

        # Stats for title
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
            ax.plot(times[sl], hub_cli_n[sl], color='red',   linestyle='--', linewidth=1.2, label='In-degree (norm)' if col == 0 else None)
            ax.plot(times[sl], hub_fit[sl],   color='green', linestyle=':',  linewidth=1.0, label='Hub fitness' if col == 0 else None)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1.05)
            if row == len(results) - 1:
                ax.set_xlabel('Time')
            if col == 0:
                ax.set_ylabel('Hub ID, in-degree, fit')
            ax.grid(alpha=0.3)
        axes[row, 0].legend(loc='upper right', fontsize=8, framealpha=0.85)

        title_text = (f'{label}  |  max_ten={max_ten}  avg_ten={avg_ten:.1f}  '
                      f'turnovers={turnovers}  avg_clients={avg_cli:.1f}')
        axes[row, 0].set_title(title_text, fontsize=10, loc='left')

    fig.suptitle('Brini Fig.6 reproduction — w58 cv=0 vs C3 γ cv=0.7 picks', fontsize=12, y=1.005)
    fig.tight_layout()
    out = os.path.join(PROJ, 'compare_fig2_brini_hub.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


def plot_fig3_equity(results_with_equity):
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ['#1a3f6b', '#27ae60', '#c0392b']
    for (label, T, equity), color in zip(results_with_equity, colors):
        ax.plot(np.arange(T), equity, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('System equity Σ E_i', fontsize=12)
    ax.set_title('System equity over time — w58 cv=0 vs C3 γ cv=0.7 picks (seed=26474)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PROJ, 'compare_fig3_equity.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


def main():
    print('Generating Figure 1 (eta-sweep, multi-seed mean) ...')
    plot_fig1_eta_sweep()

    print('\nRunning 2 single-seed sims for the C3 cv=0.7 cells (SEED=26474) ...')
    jobs = [
        ('C3 γ=0.05 cv=0.7 η=0.3 (seed=26474)', 0.6, 0.7, 0.05, 0.7, 0.3, SEED),
        ('C3 γ=0.12 cv=0.7 η=0.2 (seed=26474)', 0.6, 0.7, 0.12, 0.7, 0.2, SEED),
    ]
    with ProcessPoolExecutor(max_workers=2) as ex:
        c3_results = list(ex.map(run_one, jobs))

    # w58 cv=0 from existing dash CSV
    print('Loading w58 cv=0 from existing dash CSV ...')
    w58_label, w58_T, w58_id, w58_fit, w58_cli, w58_eq = load_w58_dash()

    # Assemble for figures 2 and 3
    fig2_data = [
        (w58_label, w58_T, w58_id, w58_fit, w58_cli),
    ]
    fig3_data = [
        (w58_label, w58_T, w58_eq),
    ]
    for label, T, hub_id, hub_fit, hub_cli, equity in c3_results:
        fig2_data.append((label, T, hub_id, hub_fit, hub_cli))
        fig3_data.append((label, T, equity))

    print('\nGenerating Figure 2 (Brini-style hub overlay) ...')
    plot_fig2_brini(fig2_data)

    print('\nGenerating Figure 3 (equity time-series) ...')
    plot_fig3_equity(fig3_data)

    print('\nDone.')


if __name__ == '__main__':
    main()
