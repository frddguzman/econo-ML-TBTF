"""§3.5.5 — Fiscal deaths vs η across canonical 3 regimes (REVISED).

Output: 3_5_5_fiscal_deaths_3regimes.png
Body figure: canonical 3 regimes only (no_tax, ex-post, ex-ante τ=1e-5).
τ-variants live in §3.5.6 levy calibration.

Source: sweep_w58_canonical_full_grid_raw.csv (375 sims).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, regime_label, asset_path, save_figure,
    setup_mpl, REGIME_COLORS, REGIME_LABEL_NICE
)

setup_mpl()
OUT = asset_path('ch3_5_claim4_eta', '3_5_5_fiscal_deaths_3regimes.png')


def main():
    raw = load_sweep('sweep_w58_canonical_full_grid_raw.csv',
                     type_map={'eta': float, 'fund_levy_rate': float,
                               'seed': int, 'fiscal_deaths': int})
    for r in raw:
        r['regime'] = regime_label(r)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for reg in ['no_tax', 'ex_post', 'ex_ante_t5']:
        sub = [r for r in raw if r['regime'] == reg]
        if not sub: continue
        etas = sorted(set(r['eta'] for r in sub))
        xs, ys, errs = [], [], []
        for eta in etas:
            cell = [r for r in sub if r['eta'] == eta]
            vals = [r['fiscal_deaths'] for r in cell]
            xs.append(eta); ys.append(np.mean(vals)); errs.append(np.std(vals))
        color = REGIME_COLORS.get(reg, 'gray')
        label = REGIME_LABEL_NICE.get(reg, reg)
        ax.plot(xs, ys, '-o', color=color, label=label, markersize=5, linewidth=1.5)
        ys_a, errs_a = np.array(ys), np.array(errs)
        ax.fill_between(xs, ys_a - errs_a, ys_a + errs_a, alpha=0.18, color=color, linewidth=0)

    ax.set_xlabel(r'Bailout coverage $\eta$')
    ax.set_ylabel('Fiscal deaths per run (5-seed mean)')
    ax.set_title(r'Fiscal deaths vs $\eta$ across canonical 3 regimes (5-seed mean $\pm$ std)')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(bottom=0)
    save_figure(fig, OUT)


if __name__ == '__main__':
    main()
