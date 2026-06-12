"""§3.1.x — Total bankruptcies vs η at 3-marker grid {0, 0.10, 0.85}.

Per Phase 2 restructure (audit LD1): η-grid locked to {0, 0.10, 0.85}.
This figure becomes the η-on-bankruptcies anchor inside the η-sweep chapter.

Output: 3_5_2_total_bk_vs_eta_3regimes_5seed.png
Body figure: canonical 3 regimes only (no_tax, ex-post, ex-ante τ=1e-5).

Source: sweep_w58_canonical_full_grid_raw.csv (375 sims, filtered to 3 etas).
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
OUT = asset_path('ch3_5_claim4_eta', '3_5_2_total_bk_vs_eta_3regimes_5seed.png')

# Locked η-grid for §3.1 figures (LD1)
ETA_GRID = [0.0, 0.10, 0.85]
ETA_TOL = 1e-9


def main():
    raw = load_sweep('sweep_w58_canonical_full_grid_raw.csv',
                     type_map={'eta': float, 'fund_levy_rate': float,
                               'seed': int, 'total_bk': int})
    for r in raw:
        r['regime'] = regime_label(r)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # Canonical 3 regimes only
    for reg in ['no_tax', 'ex_post', 'ex_ante_t5']:
        sub = [r for r in raw if r['regime'] == reg]
        if not sub: continue
        # Filter to locked η-grid {0, 0.10, 0.85} per LD1
        etas = sorted(e for e in ETA_GRID
                      if any(abs(r['eta'] - e) < ETA_TOL for r in sub))
        xs, ys, errs = [], [], []
        for eta in etas:
            cell = [r for r in sub if abs(r['eta'] - eta) < ETA_TOL]
            vals = [r['total_bk'] for r in cell]
            xs.append(eta); ys.append(np.mean(vals)); errs.append(np.std(vals))
        color = REGIME_COLORS.get(reg, 'gray')
        label = REGIME_LABEL_NICE.get(reg, reg)
        ys_a, errs_a = np.array(ys), np.array(errs)
        ax.plot(xs, ys_a, '-o', color=color, label=label, markersize=6, linewidth=1.6)
        ax.fill_between(xs, ys_a - errs_a/3, ys_a + errs_a/3, alpha=0.18, color=color, linewidth=0)

    # y-range: include error bands + light padding above/below so bands don't dominate
    all_lo, all_hi = [], []
    for reg in ['no_tax', 'ex_post', 'ex_ante_t5']:
        sub = [r for r in raw if r['regime'] == reg]
        etas = [e for e in ETA_GRID
                if any(abs(r['eta'] - e) < ETA_TOL for r in sub)]
        for eta in etas:
            cell = [r for r in sub if abs(r['eta'] - eta) < ETA_TOL]
            if cell:
                vals = [r['total_bk'] for r in cell]
                m = np.mean(vals); s = np.std(vals)
                all_lo.append(m - s); all_hi.append(m + s)
    if all_lo and all_hi:
        # 15% range padding so error bands don't visually fill the panel
        ymin = min(all_lo) - (max(all_hi) - min(all_lo)) * 0.15
        ymax = max(all_hi) + (max(all_hi) - min(all_lo)) * 0.15
    else:
        ymin, ymax = 2400, 3400
    ax.set_ylim(ymin, ymax)

    ax.axvline(0.10, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    # Annotation in upper-left corner — won't overlap with curves
    ax.text(0.02, 0.97, r'canonical $\eta^*=0.10$ (ex-post)',
            transform=ax.transAxes, fontsize=8, color='gray',
            va='top', ha='left')

    ax.set_xlabel(r'Bailout coverage $\eta$')
    ax.set_ylabel('Total bankruptcies (5-seed mean)')
    ax.set_title(r'Total bankruptcies vs $\eta$ at canonical w58 cv=0 (5-seed mean $\pm$ std, shaded band)')
    ax.set_xticks(ETA_GRID)
    ax.legend(loc='lower right', fontsize=9)
    save_figure(fig, OUT)


if __name__ == '__main__':
    main()
