"""Brini-style hub diagnostics for 3 specific bsl truncated cells the user flagged.

Cell 1 (interesting moderate-tight cap): cv=0.7, cap=1.75, floor=0.60, η=0.15  → max_ten=490±190
Cell 2 (cv=0.5 bimodal at loose cap):    cv=0.5, cap=2.50, floor=0.50, η=0.10  → max_ten=45±82, s/m=1.82
Cell 3 (cv=0.7 bimodal at loose cap):    cv=0.7, cap=2.50, floor=0.50, η=0.05  → max_ten=177±258, s/m=1.45

Each PNG: 5 seeds × 2 time panels (t=0..500, t=500..1000), Brini hub diagnostics.

Output: bsl_trunc3_brini_<label>.png × 3
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
WORKERS = 6
PROJ = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    # (label, cv, max_factor, min_factor, eta)
    ('cv07_c175_f060_e015', 0.7, 1.75, 0.60, 0.15),
    ('cv05_c250_f050_e010', 0.5, 2.50, 0.50, 0.10),
    ('cv07_c250_f050_e005', 0.7, 2.50, 0.50, 0.05),
]


def run_one(args):
    cv, max_f, min_f, eta, seed = args
    cfg = ddr.make_config(basis='equity', omega=0.50, eta=eta, regime='socialized_tax')
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    cfg['equity_heterogeneity'] = True
    cfg['equity_cv'] = cv
    cfg['equity_max_factor'] = max_f
    cfg['equity_min_factor'] = min_f
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
        'cv': cv, 'max_factor': max_f, 'min_factor': min_f, 'eta': eta, 'seed': seed, 'T': T,
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


def plot_one_cell(label, cv, max_f, min_f, eta, results_for_cell, out_path):
    rs = sorted(results_for_cell, key=lambda x: x['seed'])
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
    fig.suptitle(f'Brini-style hub diagnostics — bsl + cv={cv} + cap={max_f}, floor={min_f} + social + eta={eta}',
                 fontsize=12, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    jobs = [(cv, max_f, min_f, eta, s) for (_, cv, max_f, min_f, eta) in CELLS for s in SEEDS]
    n = len(jobs)
    print(f'Running {n} sims (3 cells x 5 seeds)...', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            results.append(r)
            if i % 5 == 0:
                print(f'  ... {i}/{n} done', flush=True)

    print('Generating PNGs...', flush=True)
    for label, cv, max_f, min_f, eta in CELLS:
        rs = [r for r in results if abs(r['cv']-cv)<1e-6 and abs(r['max_factor']-max_f)<1e-6
              and abs(r['min_factor']-min_f)<1e-6 and abs(r['eta']-eta)<1e-6]
        out_path = os.path.join(PROJ, f'bsl_trunc3_brini_{label}.png')
        plot_one_cell(label, cv, max_f, min_f, eta, rs, out_path)
        max_tens = [compute_tenure_stats(r['hub_id'])[0] for r in sorted(rs, key=lambda x: x['seed'])]
        total_bks = [r['total_bk'] for r in sorted(rs, key=lambda x: x['seed'])]
        print(f'  {label}: max_ten={max_tens}  total_bk={total_bks}  -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
