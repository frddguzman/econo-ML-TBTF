"""V2 comparison: w58 cv=0 (SEED=26474) vs C3+cv=0.7 user-picks (SEED=26466 champion).

Generates:
  Fig 1: eta-sweep MC mean +/- std (re-uses what we have but adds median to show robustness)
  Fig 2: Brini-style hub overlay at canonical champion seeds
         (w58 cv=0 SEED=26474, C3 cells SEED=26466)
  Fig 3: System equity time-series at canonical seeds
  Fig 4: Within-seed eta comparison (eta=0 vs eta* vs eta=0.85 at SEED=26466) for ONE cell
         showing the within-seed claim 3 dynamic
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
SIM = str(Path(os.environ.get(
    'TBTF_SIM_DIR',
    str(Path(__file__).resolve().parent.parent / 'Simulations'),
)))
N = 50

CHAMPION_SEED = 26466   # canonical champion seed for C3+cv=0.7 cells
W58_SEED = 26474        # dashboard convention — w58 cv=0 already non-bimodal here


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
    bk      = np.array([float(v) if v is not None else 0 for v in s.bankruptcy[:T]])
    return label, T, hub_id, hub_fit, hub_cli, equity, bk


def load_w58_dash(eta_tag):
    """Read existing dash_w58_st_<eta_tag>.csv. eta_tag in {'e0','e01','e085'}."""
    path = os.path.join(SIM, f'dash_w58_st_{eta_tag}.csv')
    hub_id, hub_fit, hub_cli, equity, bk = [], [], [], [], []
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
            try:
                bk.append(float(r['bankruptcy']))
            except (ValueError, KeyError, TypeError):
                bk.append(0.0)
    return (len(hub_id), np.array(hub_id), np.array(hub_fit),
            np.array(hub_cli), np.array(equity), np.array(bk))


# ============================================================================
# Figure 1 — eta-sweep with mean and median bands
# ============================================================================

def load_w58_eta_sweep_with_seeds():
    path = os.path.join(SIM, 'thesis_lehman_w58.csv')
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r['fiscal_regime'] == 'socialized_tax':
                rows.append(r)
    etas = sorted(set(float(r['eta']) for r in rows))
    per_eta_seeds = {}
    for eta in etas:
        bks = [int(r['total_bk']) for r in rows if abs(float(r['eta']) - eta) < 1e-6]
        per_eta_seeds[eta] = bks
    return etas, per_eta_seeds


def load_c3_eta_sweep_with_seeds(gamma, cv):
    path = os.path.join(PROJ, 'sweep_c3_hetero_full_eta_raw.csv')
    rows = []
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            if (abs(float(r['gamma']) - gamma) < 1e-6
                and abs(float(r['cv']) - cv) < 1e-6
                and r['fiscal_regime'] == 'socialized_tax'):
                rows.append(r)
    etas = sorted(set(float(r['eta']) for r in rows))
    per_eta_seeds = {}
    for eta in etas:
        bks = [int(r['total_bk']) for r in rows if abs(float(r['eta']) - eta) < 1e-6]
        per_eta_seeds[eta] = bks
    return etas, per_eta_seeds


def stats_summary(values):
    if not values: return 0, 0, 0, 0, 0
    n = len(values)
    m = sum(values) / n
    s = (sum((x - m) ** 2 for x in values) / max(1, n - 1)) ** 0.5 if n > 1 else 0
    sv = sorted(values)
    median = sv[n // 2] if n % 2 == 1 else 0.5 * (sv[n // 2 - 1] + sv[n // 2])
    return m, s, median, sv[0], sv[-1]


def plot_fig1_eta_sweep_with_median():
    w58_etas, w58_per_eta = load_w58_eta_sweep_with_seeds()
    g05_etas, g05_per_eta = load_c3_eta_sweep_with_seeds(0.05, 0.7)
    g12_etas, g12_per_eta = load_c3_eta_sweep_with_seeds(0.12, 0.7)

    fig, ax = plt.subplots(figsize=(11, 6))

    for etas, per_eta, color, label, marker, eta_star in [
        (w58_etas, w58_per_eta, '#1a3f6b', 'w58 cv=0',                's', 0.1),
        (g05_etas, g05_per_eta, '#27ae60', 'C3 γ=0.05 cv=0.7',         'o', 0.3),
        (g12_etas, g12_per_eta, '#c0392b', 'C3 γ=0.12 cv=0.7',         '^', 0.2),
    ]:
        means, medians, mins, maxs = [], [], [], []
        for eta in etas:
            m, s, med, lo, hi = stats_summary(per_eta[eta])
            means.append(m); medians.append(med); mins.append(lo); maxs.append(hi)
        ax.fill_between(etas, mins, maxs, color=color, alpha=0.15, label=f'{label} min-max range')
        ax.plot(etas, means, color=color, marker=marker, markersize=6, linewidth=2,
                label=f'{label} 5-seed mean (η*={eta_star})')
        ax.plot(etas, medians, color=color, linestyle='--', linewidth=1.2, alpha=0.8,
                label=f'{label} median')
        idx = etas.index(eta_star)
        ax.scatter([eta_star], [means[idx]], s=220, facecolors='none',
                   edgecolors=color, linewidths=2.5, zorder=6)

    ax.set_xlabel('η (bailout coverage)', fontsize=12)
    ax.set_ylabel('Total bankruptcies (5 seeds)', fontsize=12)
    ax.set_title('Social regime η-sweep: mean (solid), median (dashed), min-max range (band)',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8.5, ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PROJ, 'compare_v2_fig1_eta_sweep_mean_median.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


# ============================================================================
# Figure 2 — Brini-style hub overlay at canonical seeds
# ============================================================================

def plot_fig2_brini(results):
    """3 stacked cells × 2 columns (t=0..500, t=500..1000)."""
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 3.8 * len(results)),
                             sharey='row')
    if len(results) == 1:
        axes = axes.reshape(1, 2)

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
                      f'turnovers={turnovers}  avg_cli={avg_cli:.1f}')
        axes[row, 0].set_title(title_text, fontsize=10, loc='left')

    fig.suptitle('Brini Fig.6 reproduction at canonical seeds (champion-seed for C3 cells)',
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out = os.path.join(PROJ, 'compare_v2_fig2_brini_champion.png')
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
    ax.set_title('System equity over time at canonical seeds (champion-seed for C3 cells)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PROJ, 'compare_v2_fig3_equity.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


# ============================================================================
# Figure 4 — Within-seed eta comparison (eta=0 vs eta* vs eta=0.85)
# ============================================================================

def plot_fig4_within_seed_eta(c3_g12_results, w58_data_per_eta):
    """For each cell, 3 sub-panels showing total_bk per period at eta=0 / eta=eta* / eta=0.85.
    Cumulative bk over time. Demonstrates within-seed claim 3 effect.
    Top row: w58 cv=0 (SEED=26474, eta in {0, 0.1, 0.85}) — from dashboard CSVs
    Bottom row: C3 g=0.12 cv=0.7 SEED=26466 (eta in {0, 0.2, 0.85}) — fresh sims
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Row 0: w58
    ax = axes[0]
    eta_tags = [('e0', 0.0, '#3498db'), ('e01', 0.1, '#1a3f6b'), ('e085', 0.85, '#85929e')]
    for tag, eta_v, color in eta_tags:
        T, _hub, _fit, _cli, _eq, bk = load_w58_dash(tag)
        cum_bk = np.cumsum(bk)
        ax.plot(np.arange(T), cum_bk, label=f'η={eta_v}', color=color, linewidth=1.8)
    ax.set_ylabel('Cumulative bankruptcies', fontsize=11)
    ax.set_title('w58 cv=0 (SEED=26474) — within-seed η comparison: η=0 vs η*=0.1 vs η=0.85', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)

    # Row 1: c3 g=0.12 cv=0.7 SEED=26466
    ax = axes[1]
    colors = {'0.0': '#27ae60', '0.2': '#1a5c2a', '0.85': '#85929e'}
    for label, T, _hub, _fit, _cli, _eq, bk in c3_g12_results:
        eta_str = label.split('η=')[1].split(' ')[0]
        eta_v = float(eta_str)
        cum_bk = np.cumsum(bk)
        ax.plot(np.arange(T), cum_bk, label=f'η={eta_v}', color=colors.get(eta_str, '#888'), linewidth=1.8)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Cumulative bankruptcies', fontsize=11)
    ax.set_title('C3 γ=0.12 cv=0.7 (SEED=26466 champion) — within-seed η comparison: η=0 vs η*=0.2 vs η=0.85', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(PROJ, 'compare_v2_fig4_within_seed_eta.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


def main():
    print('Generating Figure 1 (eta-sweep with mean and median) ...')
    plot_fig1_eta_sweep_with_median()

    print('\nRunning sims at champion seed (26466) for C3 cells, plus the within-seed eta comparison ...')
    jobs = [
        # Champion-seed Brini reproduction (eta=eta*)
        (f'C3 γ=0.05 cv=0.7 η=0.3 (seed=26466)', 0.6, 0.7, 0.05, 0.7, 0.3, CHAMPION_SEED),
        (f'C3 γ=0.12 cv=0.7 η=0.2 (seed=26466)', 0.6, 0.7, 0.12, 0.7, 0.2, CHAMPION_SEED),
        # Within-seed eta comparison at C3 g=0.12 cv=0.7 (champion seed)
        (f'C3 γ=0.12 cv=0.7 η=0.0 (seed=26466)', 0.6, 0.7, 0.12, 0.7, 0.0, CHAMPION_SEED),
        (f'C3 γ=0.12 cv=0.7 η=0.85 (seed=26466)', 0.6, 0.7, 0.12, 0.7, 0.85, CHAMPION_SEED),
    ]
    with ProcessPoolExecutor(max_workers=6) as ex:
        all_results = list(ex.map(run_one, jobs))

    # Index by label suffix
    by_label = {r[0]: r for r in all_results}

    # Figures 2 + 3: w58 + 2 C3 cells at champion seed
    print('Loading w58 cv=0 from existing dash CSV (eta=0.1) ...')
    w58_T, w58_id, w58_fit, w58_cli, w58_eq, w58_bk = load_w58_dash('e01')
    fig2_data = [
        (f'w58 cv=0 η=0.1 (SEED={W58_SEED})', w58_T, w58_id, w58_fit, w58_cli),
    ]
    fig3_data = [
        (f'w58 cv=0 η=0.1 (SEED={W58_SEED})', w58_T, w58_eq),
    ]
    for label_key in [f'C3 γ=0.05 cv=0.7 η=0.3 (seed=26466)',
                      f'C3 γ=0.12 cv=0.7 η=0.2 (seed=26466)']:
        r = by_label[label_key]
        fig2_data.append((r[0], r[1], r[2], r[3], r[4]))
        fig3_data.append((r[0], r[1], r[5]))

    print('\nGenerating Figure 2 (Brini-style at champion seed) ...')
    plot_fig2_brini(fig2_data)

    print('\nGenerating Figure 3 (equity at champion seed) ...')
    plot_fig3_equity(fig3_data)

    # Figure 4: within-seed eta comparison at C3 g=0.12 cv=0.7
    c3_g12_results = [
        by_label[f'C3 γ=0.12 cv=0.7 η=0.0 (seed=26466)'],
        by_label[f'C3 γ=0.12 cv=0.7 η=0.2 (seed=26466)'],
        by_label[f'C3 γ=0.12 cv=0.7 η=0.85 (seed=26466)'],
    ]
    print('\nGenerating Figure 4 (within-seed eta comparison) ...')
    plot_fig4_within_seed_eta(c3_g12_results, None)

    print('\nDone.')


if __name__ == '__main__':
    main()
