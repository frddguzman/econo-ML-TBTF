"""§3.1.7 — ω-section: topology dichotomy across calibration ω (USER L742 thesis content).

Outputs:
- 3_1_7_omega_dichotomy_map.png    (ω vs total_bk + max_ten + classification)
- 3_1_7_omega_brini_grid.png       (4 cells × Brini hub diagnostic)

Cells: w50 (ω=0.50, bsl), w53 (ω=0.53, A), w55 (ω=0.55), w58 (ω=0.58, canonical)
All at cv=0, ex-post regime, η=0.10.

Source for map: dashboard data + canonical full grid for w58
Source for Brini grid: dash_*_st_e01.csv at SEED=26474
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import asset_path, save_figure, setup_mpl, COLORS

setup_mpl()
OUT_MAP = asset_path('ch3_1_topology', '3_1_7_omega_dichotomy_map.png')
OUT_BRINI = asset_path('ch3_1_topology', '3_1_7_omega_brini_grid.png')

CANDS = [
    ('bsl', 'w50', 0.50, '../Simulations/dash_bsl_st_e01.csv'),
    ('a',   'w53', 0.53, '../Simulations/dash_a_st_e01.csv'),
    ('w55', 'w55', 0.55, '../Simulations/dash_w55_st_e01.csv'),
    ('w58', 'w58 (canonical)', 0.58, '../Simulations/dash_w58_st_e01.csv'),
]


def load_dash_summary(path):
    """Compute summary stats from a dash CSV: total_bk (last), max_ten, max_clients."""
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    if not rows:
        return None
    try:
        # bankruptcy column is per-period; sum across all rows gives cumulative
        total_bk = sum(float(r.get('bankruptcy', 0) or 0) for r in rows)
        # Compute hub run-length from best_lender column
        best_lender = []
        for r in rows:
            try:
                v = float(r['best_lender'])
                if v >= 0: best_lender.append(int(v))
            except (ValueError, TypeError):
                continue
        # Run-length encoding
        if best_lender:
            runs = []
            prev, rl = best_lender[0], 1
            for k in best_lender[1:]:
                if k == prev: rl += 1
                else: runs.append(rl); rl = 1; prev = k
            runs.append(rl)
            max_ten = max(runs) if runs else 0
            avg_ten = np.mean(runs) if runs else 0
        else:
            max_ten = 0; avg_ten = 0
        # Clients
        clients = [float(r['best_lender_clients']) for r in rows
                   if r.get('best_lender_clients') and float(r['best_lender_clients']) >= 0]
        max_cli = max(clients) if clients else 0
        return {'total_bk': total_bk, 'max_ten': max_ten, 'avg_ten': avg_ten,
                'max_cli': max_cli}
    except (ValueError, TypeError, KeyError):
        return None


def render_map():
    """Bar chart: ω vs total_bk + max_ten + max_cli."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    omegas = [c[2] for c in CANDS]
    labels = [c[1] for c in CANDS]
    summaries = [load_dash_summary(c[3]) for c in CANDS]

    metrics = [
        ('total_bk', 'Total bankruptcies', COLORS['cBlue']),
        ('max_ten',  'Max hub tenure (periods)', COLORS['cRed']),
        ('max_cli',  'Max hub clients', COLORS['cGreen']),
    ]
    for ax, (key, title, color) in zip(axes, metrics):
        vals = [(s[key] if s else 0) for s in summaries]
        bars = ax.bar(range(len(CANDS)), vals, color=color, width=0.6,
                      edgecolor='white', linewidth=0.5)
        # Highlight w58 canonical
        bars[3].set_edgecolor('black')
        bars[3].set_linewidth(2)
        ax.set_xticks(range(len(CANDS)))
        ax.set_xticklabels(labels, rotation=15)
        ax.set_xlabel(r'$\omega$ cell (cv=0, ex-post tax, $\eta=0.10$)')
        ax.set_ylabel(title, fontsize=9)
        ax.set_title(title, fontsize=10)

    fig.suptitle(r'$\omega$ topology dichotomy at cv=0 — w58 = unique stable-hub calibration',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, OUT_MAP)


def render_brini_grid():
    """4-cell grid: Brini-style hub_id + clients per ω cell."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    for ax, (cand_tag, label, omega, path) in zip(axes, CANDS):
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f'NO DATA\n{cand_tag}', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=9)
            ax.set_title(label, fontsize=10)
            continue
        rows = list(csv.DictReader(open(path, encoding='utf-8')))
        t = np.array([float(r['time']) for r in rows])
        bl = np.array([float(r['best_lender']) for r in rows])
        bl = np.where(bl < 0, np.nan, bl)
        ax.plot(t, bl, color=COLORS['cBlue'], linewidth=0.5, drawstyle='steps-post')
        ax.set_xlabel('time')
        ax.set_ylabel('Hub id') if ax is axes[0] else None
        ax.set_title(label + (' ★' if cand_tag == 'w58' else ''), fontsize=10)
        ax.set_ylim(0, 50)

    fig.suptitle(r'Hub identity over time across $\omega$ cells (cv=0, ex-post, $\eta=0.10$, SEED=26474)',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, OUT_BRINI)


def main():
    render_map()
    render_brini_grid()


if __name__ == '__main__':
    main()
