"""§3.4.4 — ρ-peak persistence across α — REVISED to heatmap.

Per USER 2026-05-09: the line-overlay version didn't communicate the message clearly.
Heatmap layout: x=ρ, y=α, color=total_bk. The ρ-peak at 0.7 should appear as a hot
vertical band; α-modulation appears as vertical gradient.

3 regimes × 3 etas = 9 panels in a 3×3 grid.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, asset_path, save_figure, setup_mpl
)

setup_mpl()
OUT = asset_path('ch3_4_claim3_rho', '3_4_4_rho_peak_persistence_alpha.png')

ETAS = [0.10, 0.50, 0.85]


def load_combined():
    rows = []
    for r in load_sweep('sweep_w58_alpha_rho_raw.csv', type_map={
        'alpha_collateral': float, 'rho': float, 'eta': float, 'seed': int,
        'total_bk': int}):
        if r['fiscal_regime'] == 'none':
            r['regime'] = 'no_tax'
        elif r['fiscal_regime'] == 'socialized_tax':
            r['regime'] = 'ex_post'
        else:
            continue
        rows.append(r)
    for r in load_sweep('sweep_w58_alpha_rho_fund_t5_raw.csv', type_map={
        'alpha_collateral': float, 'rho': float, 'eta': float, 'seed': int,
        'total_bk': int}):
        r['regime'] = 'ex_ante_t5'
        rows.append(r)
    return rows


def main():
    rows = load_combined()
    alphas = sorted(set(r['alpha_collateral'] for r in rows))
    rhos = sorted(set(r['rho'] for r in rows))
    REGIMES = ['no_tax', 'ex_post', 'ex_ante_t5']
    REG_NICE = {'no_tax': 'No tax', 'ex_post': 'Ex-post tax',
                'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$'}

    # Compute global vmin/vmax for unified color scale across panels
    all_vals = []
    cells = {}
    for reg in REGIMES:
        for eta in ETAS:
            grid = np.full((len(alphas), len(rhos)), np.nan)
            for i, a in enumerate(alphas):
                for j, rh in enumerate(rhos):
                    vals = [r['total_bk'] for r in rows
                            if r['regime'] == reg and r['eta'] == eta
                            and r['alpha_collateral'] == a and r['rho'] == rh]
                    if vals:
                        grid[i, j] = np.mean(vals)
                        all_vals.append(np.mean(vals))
            cells[(reg, eta)] = grid

    vmin, vmax = (np.nanpercentile(all_vals, 5),
                  np.nanpercentile(all_vals, 95)) if all_vals else (0, 1)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for i, reg in enumerate(REGIMES):
        for j, eta in enumerate(ETAS):
            ax = axes[i, j]
            grid = cells.get((reg, eta), np.full((len(alphas), len(rhos)), np.nan))
            im = ax.imshow(grid, aspect='auto', origin='lower', cmap='YlOrRd',
                           extent=[rhos[0] - 0.05, rhos[-1] + 0.05,
                                   alphas[0] - 0.01, alphas[-1] + 0.01],
                           vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.axvline(0.70, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
            # Annotate cell values
            for ii, a in enumerate(alphas):
                for jj, rh in enumerate(rhos):
                    val = grid[ii, jj]
                    if not np.isnan(val):
                        # text color: black on light, white on dark
                        norm = (val - vmin) / max(vmax - vmin, 1)
                        text_color = 'white' if norm > 0.6 else 'black'
                        ax.text(rh, a, f'{val:.0f}', ha='center', va='center',
                                fontsize=7, color=text_color)
            if i == 0:
                ax.set_title(rf'$\eta = {eta:.2f}$', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'{REG_NICE[reg]}\n' + r'$\alpha$', fontsize=9)
            if i == 2:
                ax.set_xlabel(r'$\rho$')
            ax.set_xticks(rhos)
            ax.set_yticks(alphas)
            ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, location='right')
    cbar.set_label('Total bankruptcies (5-seed mean)', fontsize=9)
    fig.suptitle(r'$\rho$-peak persistence across $\alpha$ — heatmap of total bankruptcies '
                 r'(rows: regime; columns: $\eta$; vertical line: $\rho=0.70$)',
                 fontsize=11, y=0.99)
    save_figure(fig, OUT)


if __name__ == '__main__':
    main()
