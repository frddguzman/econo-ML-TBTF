"""§3.1.6 — Hub turnover causality (Prof Request B; bonus).

Outputs:
- 3_1_6_hub_turnover_causality.png
- 3_1_6_hub_turnover_causality.tex

Distinguishes hub-changes by cause:
- "fitness drop" (previous hub still alive, just lost attractiveness)
- "death" (previous hub bankrupt)
Aggregated per (regime, η) using `previous_hub_alive` column from dashboard CSVs.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    asset_path, save_figure, setup_mpl, COLORS, write_booktabs_table
)

setup_mpl()
OUT_FIG = asset_path('ch3_1_topology', '3_1_6_hub_turnover_causality.png')
OUT_TBL = asset_path('ch3_1_topology', '3_1_6_hub_turnover_causality.tex')

REGIMES = [
    ('nt',    'No tax'),
    ('st',    'Ex-post tax'),
    ('t5_rf', r'Ex-ante $\tau{=}10^{-5}$'),
]
ETAS = [('e0', 0.0), ('e01', 0.10), ('e05', 0.50), ('e085', 0.85)]


def load_causality(regime_tag, eta_tag):
    """Count hub-change events by cause from previous_hub_alive column."""
    if regime_tag == 't5_rf':
        path = f'../Simulations/dash_w58_t5_rf_{eta_tag}.csv'
    else:
        path = f'../Simulations/dash_w58_{regime_tag}_{eta_tag}.csv'
    if not os.path.exists(path):
        return None, None
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    fitness_drop = 0
    death = 0
    prev_hub = None
    for r in rows:
        try:
            cur = float(r['best_lender'])
            prev_alive = float(r.get('previous_hub_alive', -1))
        except (ValueError, TypeError):
            continue
        if cur < 0: continue
        if prev_hub is not None and cur != prev_hub:
            # hub changed
            if prev_alive > 0:
                fitness_drop += 1
            elif prev_alive == 0:
                death += 1
        prev_hub = cur
    return fitness_drop, death


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    table_rows = []
    for ax, (reg_tag, label) in zip(axes, REGIMES):
        fd_vals = []
        d_vals = []
        eta_vals_for_plot = []
        for eta_tag, eta_v in ETAS:
            fd, d = load_causality(reg_tag, eta_tag)
            if fd is None:
                table_rows.append([label if eta_v == 0 else '', f'{eta_v:.2f}', '--', '--'])
                continue
            total = fd + d
            fd_pct = 100 * fd / total if total > 0 else 0
            d_pct = 100 * d / total if total > 0 else 0
            fd_vals.append(fd); d_vals.append(d)
            eta_vals_for_plot.append(eta_v)
            table_rows.append([label if eta_v == 0 else '', f'{eta_v:.2f}',
                               f'{fd} ({fd_pct:.0f}\\%)', f'{d} ({d_pct:.0f}\\%)'])

        if eta_vals_for_plot:
            x = np.arange(len(eta_vals_for_plot))
            totals = np.array([fd + d for fd, d in zip(fd_vals, d_vals)])
            fd_arr = np.array(fd_vals)
            d_arr = np.array(d_vals)
            with np.errstate(divide='ignore', invalid='ignore'):
                fd_share = np.where(totals > 0, fd_arr / totals, 0)
                d_share = np.where(totals > 0, d_arr / totals, 0)
            ax.bar(x, fd_share, color=COLORS['cBlue'], label='fitness drop', width=0.6)
            ax.bar(x, d_share, bottom=fd_share, color=COLORS['cRed'],
                   label='death', width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{e:.2f}' for e in eta_vals_for_plot])
        ax.set_ylim(0, 1.18)  # leave headroom for legend
        ax.set_xlabel(r'$\eta$')
        if ax is axes[0]:
            ax.set_ylabel('Share of hub-change events')
        ax.set_title(label, fontsize=10)
    # Single shared legend below, no overlay
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS['cBlue']),
               plt.Rectangle((0, 0), 1, 1, color=COLORS['cRed'])]
    fig.legend(handles, ['Fitness drop (previous hub still alive)', 'Death (previous hub bankrupt)'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    fig.suptitle('Hub-change causality at canonical w58 cv=0, single-seed (SEED=26474)',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save_figure(fig, OUT_FIG)

    write_booktabs_table(
        rows=table_rows, columns=[0, 1, 2, 3],
        col_headers=['Regime', r'$\eta$', 'Fitness drop', 'Death'],
        path=OUT_TBL, column_format='lcrr',
        caption=(r'Hub-change attribution by cause at canonical w58 cv=0, SEED=26474. '
                 r'Fitness drop = previous hub still alive but lost attractiveness; '
                 r'Death = previous hub bankrupt. Prof Request B (bonus).'),
        label='tab:hub_turnover_causality'
    )


if __name__ == '__main__':
    main()
