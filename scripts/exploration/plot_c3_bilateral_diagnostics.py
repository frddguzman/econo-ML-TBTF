"""Brini figures + total_bk-vs-eta sweep plot for C3 + tier_init + bilateral + mult=1.75.

Outputs:
  c3_bilateral_brini_per_seed.png  -- 5 rows (seeds) x 2 cols (t=[0,500], [500,1000])
                                      Each panel: hub_id (black, normalised /N),
                                      hub_clients (red dashed, normalised /max),
                                      hub_fitness (green dotted)
  c3_bilateral_total_bk_vs_eta.png -- single plot, mean total_bk per eta with error bars,
                                      eta* marked

Cell: C3 (mu=0.60, omega=0.70, gamma=0.12) + tier_init + fitness_basis='bilateral'
      + n_big=3 + E_big_multiplier=1.75
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
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
ETAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ETA_FOR_BRINI = 0.1
MULT = 1.75
N_BIG = 3
N = 50
WORKERS = 6

PROJ = os.path.dirname(os.path.abspath(__file__))


def run_one(args):
    eta, seed = args
    cfg = ddr.make_config(basis='bilateral', omega=0.70, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.60
    cfg['gamma_capital'] = 0.12
    cfg['tier_init'] = True
    cfg['n_big'] = N_BIG
    cfg['E_big_multiplier'] = MULT
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


def plot_brini(results_eta01, out_path):
    """5 seeds, each row: 2 panels (t=0..500 and t=500..1000), 3 traces per panel."""
    fig, axes = plt.subplots(len(results_eta01), 2, figsize=(13, 3.8 * len(results_eta01)), sharey='row')
    if len(results_eta01) == 1:
        axes = axes.reshape(1, 2)
    for row, r in enumerate(results_eta01):
        T = r['T']
        hub_id  = r['hub_id']
        hub_fit = r['hub_fit']
        hub_cli = r['hub_cli']
        cli_max = max(20.0, float(np.nanmax(hub_cli)) if not np.all(np.isnan(hub_cli)) else 20.0)
        hub_id_n  = hub_id / N
        hub_cli_n = hub_cli / cli_max
        max_ten, avg_ten, turnovers = compute_tenure_stats(hub_id)
        avg_cli = float(np.nanmean(hub_cli)) if not np.all(np.isnan(hub_cli)) else 0
        times = np.arange(T)
        for col, (lo, hi) in enumerate([(0, 500), (500, 1000)]):
            ax = axes[row, col]
            sl = slice(lo, hi)
            ax.plot(times[sl], hub_id_n[sl],  color='black', linewidth=1.0, label='Hub ID (norm /N)' if col == 0 else None)
            ax.plot(times[sl], hub_cli_n[sl], color='red',   linestyle='--', linewidth=1.2, label='Clients (norm /max)' if col == 0 else None)
            ax.plot(times[sl], hub_fit[sl],   color='green', linestyle=':', linewidth=1.0, label='Hub fitness' if col == 0 else None)
            ax.set_xlim(lo, hi); ax.set_ylim(0, 1.05)
            if row == len(results_eta01) - 1: ax.set_xlabel('Time')
            if col == 0: ax.set_ylabel('Hub ID, in-degree, fit')
            ax.grid(alpha=0.3)
        axes[row, 0].legend(loc='upper right', fontsize=8, framealpha=0.85)
        title_text = (f'seed={r["seed"]}  |  max_ten={max_ten}  avg_ten={avg_ten:.1f}  '
                      f'turnovers={turnovers}  avg_cli={avg_cli:.1f}  total_bk={r["total_bk"]}')
        axes[row, 0].set_title(title_text, fontsize=10, loc='left')
    fig.suptitle(f'Brini-style hub diagnostics — C3 + tier_init + bilateral + mult={MULT} + eta={ETA_FOR_BRINI}',
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}', flush=True)


def plot_eta_sweep(results_all, out_path):
    """Mean total_bk vs eta with error bars across 5 seeds."""
    means = []
    stds = []
    for eta in ETAS:
        vals = [r['total_bk'] for r in results_all if abs(r['eta'] - eta) < 1e-9]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
    bk0 = means[0]
    eta_star_idx = int(np.argmin(means))
    eta_star = ETAS[eta_star_idx]
    bk_star = means[eta_star_idx]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(ETAS, means, yerr=stds, marker='o', capsize=4, linewidth=1.5,
                color='#1a3f6b', label='mean total_bk ± std (5 seeds)')
    ax.axhline(bk0, color='gray', linestyle=':', linewidth=1, label=f'η=0 baseline ({bk0:.0f})')
    ax.axvline(eta_star, color='red', linestyle='--', linewidth=1, alpha=0.6,
               label=f'η*={eta_star} (min total_bk={bk_star:.0f}, Δ={bk_star-bk0:+.0f})')
    # Annotate every point with its mean
    for eta, m in zip(ETAS, means):
        ax.annotate(f'{m:.0f}', (eta, m), textcoords='offset points', xytext=(0, 8),
                    fontsize=8, ha='center')
    ax.set_xlabel('η (bailout coverage)', fontsize=12)
    ax.set_ylabel('total_bk (mean across 5 seeds, ± std)', fontsize=12)
    ax.set_title(f'C3 + tier_init + bilateral + mult={MULT}: total_bk vs η\n'
                 f'(claim 3 verdict: {"PRESERVED" if bk_star < bk0 else "BROKEN"} | Δ={bk_star-bk0:+.0f})',
                 fontsize=12)
    ax.set_xlim(-0.03, 0.95)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}', flush=True)


def main():
    jobs = [(e, s) for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'Running {n} sims (5 seeds × 12 etas) at C3 + tier_init + bilateral + mult={MULT}...', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            results.append(r)
            if i % 15 == 0:
                print(f'  ... {i}/{n} done', flush=True)
    print(f'All sims complete. Generating figures...', flush=True)

    # Brini figs at eta = ETA_FOR_BRINI (the optimum)
    eta01_results = sorted([r for r in results if abs(r['eta'] - ETA_FOR_BRINI) < 1e-9],
                           key=lambda x: x['seed'])
    plot_brini(eta01_results, os.path.join(PROJ, 'c3_bilateral_brini_per_seed.png'))
    plot_eta_sweep(results, os.path.join(PROJ, 'c3_bilateral_total_bk_vs_eta.png'))


if __name__ == '__main__':
    main()
