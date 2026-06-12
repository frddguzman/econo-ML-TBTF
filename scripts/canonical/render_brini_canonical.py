"""§3.1.1 — Brini Fig.6 hub diagnostic — REVISED to match Brini-Tedeschi-Tantari 2023 Fig.6.

Single overlay per panel: hub_id (black solid), hub_fitness (green dotted),
clients (red dashed) — all normalized to [0, 1]. Time split 0-500 / 500-1000 per regime.

Per-eta separate figures (4 total: e0, e01, e05, e085).
Each figure: 3 regimes × 2 time-windows = 6 panels in a 3-row × 2-col grid.

Single-seed SEED=26474 (dashboard-sourced).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import csv
import matplotlib.pyplot as plt
from thesis_render_utils import asset_path, save_figure, setup_mpl, COLORS

setup_mpl()

REGIMES = [
    ('nt',    'No tax'),
    ('st',    'Ex-post tax'),
    ('t5_rf', r'Ex-ante $\tau{=}10^{-5}$'),
]
ETAS = [
    ('e0',   r'$\eta = 0$',       '0'),
    ('e01',  r'$\eta = 0.10$',    '01'),
    ('e05',  r'$\eta = 0.50$',    '05'),
    ('e085', r'$\eta = 0.85$',    '085'),
]


def load_dash(regime_tag, eta_tag):
    if regime_tag == 't5_rf':
        path = f'../Simulations/dash_w58_t5_rf_{eta_tag}.csv'
    else:
        path = f'../Simulations/dash_w58_{regime_tag}_{eta_tag}.csv'
    if not os.path.exists(path):
        return None
    return list(csv.DictReader(open(path, encoding='utf-8')))


def render_one_eta(eta_tag, eta_label):
    """Brini Fig.6 layout: 3 regimes × 2 time windows (0-500, 500-1000).
    Single y-axis [0, 1] per panel; 3 series overlaid.
    """
    fig, axes = plt.subplots(len(REGIMES), 2, figsize=(13, 8),
                             sharey=True,
                             gridspec_kw={'hspace': 0.30, 'wspace': 0.05})
    N = 50

    for i, (reg_tag, reg_label) in enumerate(REGIMES):
        rows = load_dash(reg_tag, eta_tag)
        for col, (t_lo, t_hi) in enumerate([(0, 500), (500, 1000)]):
            ax = axes[i, col]
            if rows is None:
                ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                        transform=ax.transAxes, color='gray', fontsize=9)
                continue
            t = np.array([float(r['time']) for r in rows])
            bl = np.array([float(r['best_lender']) for r in rows])
            bl = np.where(bl < 0, np.nan, bl)
            fit = np.array([float(r['best_lender_fitness']) for r in rows])
            fit = np.where(fit < 0, np.nan, fit)
            cli = np.array([float(r['best_lender_clients']) for r in rows])
            cli = np.where(cli < 0, np.nan, cli)

            mask = (t >= t_lo) & (t <= t_hi)
            t_w = t[mask]
            # Normalize all to [0, 1]:
            #  hub_id: divide by N
            #  fitness: already [0, 1]
            #  clients: divide by N (max possible)
            hub_id_norm = bl[mask] / N
            cli_norm = cli[mask] / N

            # Overlay (Brini convention)
            ax.plot(t_w, hub_id_norm, color='black', linewidth=0.7,
                    drawstyle='steps-post', label='Hub ID' if (i==0 and col==0) else None)
            ax.plot(t_w, fit[mask], color=COLORS['cGreen'], linewidth=0.6,
                    linestyle=':', alpha=0.85,
                    label='Fitness' if (i==0 and col==0) else None)
            ax.plot(t_w, cli_norm, color=COLORS['cRed'], linewidth=0.7,
                    linestyle='--', alpha=0.85,
                    label='Clients (in-degree)' if (i==0 and col==0) else None)

            ax.set_xlim(t_lo, t_hi)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Time' if i == len(REGIMES) - 1 else '', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{reg_label}\n(normalized)', fontsize=9)
            ax.tick_params(labelsize=8)

    # Single shared legend (top)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=10, frameon=False)

    fig.suptitle(f'Hub formation diagnostic (Brini Fig.6 layout) — w58 cv=0, '
                 f'{eta_label}, SEED=26474',
                 fontsize=11, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = asset_path('ch3_1_topology', f'3_1_1_brini_canonical_{eta_tag}.png')
    save_figure(fig, out)


def main():
    for eta_tag, eta_label, _ in ETAS:
        render_one_eta(eta_tag, eta_label)


if __name__ == '__main__':
    main()
