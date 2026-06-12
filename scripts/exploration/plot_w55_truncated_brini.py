"""Brini-style figs for w55 + truncated lognormal (cap=2.5, floor=0.5) cv=0.7 across all etas.

Goal: visually confirm Approach B candidate (w55 cv=0.7 truncated) shows clean unimodal hub
dynamics with no star foreclosure. Compare directly to bsl_hetero_w45_brini_e*.png set.
Output: 12 PNGs w55_trunc_cv07_brini_e{XX}.png at all etas in the sweep.
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
ETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OMEGA = 0.55
MU = 0.70
GAMMA = 0.10
CV = 0.7
MAX_FACTOR = 2.5
MIN_FACTOR = 0.5
N = 50
WORKERS = 6

PROJ = os.path.dirname(os.path.abspath(__file__))


def run_one(args):
    eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime='socialized_tax')
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = CV
    cfg['equity_max_factor'] = MAX_FACTOR
    cfg['equity_min_factor'] = MIN_FACTOR
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


def plot_one_eta(eta, results_for_eta, out_path):
    rs = sorted(results_for_eta, key=lambda x: x['seed'])
    fig, axes = plt.subplots(len(rs), 2, figsize=(13, 3.8 * len(rs)), sharey='row')
    if len(rs) == 1:
        axes = axes.reshape(1, 2)
    for row, r in enumerate(rs):
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
            if row == len(rs) - 1: ax.set_xlabel('Time')
            if col == 0: ax.set_ylabel('Hub ID, in-degree, fit')
            ax.grid(alpha=0.3)
        axes[row, 0].legend(loc='upper right', fontsize=8, framealpha=0.85)
        title_text = (f'seed={r["seed"]}  |  max_ten={max_ten}  avg_ten={avg_ten:.1f}  '
                      f'turnovers={turnovers}  avg_cli={avg_cli:.1f}  total_bk={r["total_bk"]}')
        axes[row, 0].set_title(title_text, fontsize=10, loc='left')
    fig.suptitle(f'Brini-style hub diagnostics — w55 + truncated lognormal (cap={MAX_FACTOR}, floor={MIN_FACTOR}) + cv={CV} + eta={eta}',
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    jobs = [(e, s) for e in ETAS for s in SEEDS]
    n = len(jobs)
    print(f'Running {n} sims at w55 + truncated lognormal cv={CV}...', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            results.append(r)
            if i % 10 == 0:
                print(f'  ... {i}/{n} done', flush=True)

    print(f'Generating {len(ETAS)} Brini PNGs...', flush=True)
    for eta in ETAS:
        rs = [r for r in results if abs(r['eta'] - eta) < 1e-9]
        eta_int = int(round(eta * 10))
        out_path = os.path.join(PROJ, f'w55_trunc_cv07_brini_e{eta_int:02d}.png')
        plot_one_eta(eta, rs, out_path)
        max_tens = []
        for r in sorted(rs, key=lambda x: x['seed']):
            mt, _, _ = compute_tenure_stats(r['hub_id'])
            max_tens.append(mt)
        total_bks = [r['total_bk'] for r in sorted(rs, key=lambda x: x['seed'])]
        print(f'  eta={eta}: max_ten={max_tens}  total_bk={total_bks}  -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
