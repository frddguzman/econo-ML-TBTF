"""Smoking gun: total_bk vs eta — comparing canonical tau=1e-5 vs proposed tau=3e-5
both at ex-ante regime, and with no-tax + ex-post tax for full body context.

Output: tau_compare_eta_sweep.png (smoking-gun decision figure for user)
Sources:
- sweep_w58_canonical_full_grid_raw.csv: no_tax + ex_post + ex_ante τ={1e-4, 1e-5, 1e-6}
- smoke_eta_sweep_tau3e5_raw.csv: ex-ante τ=3e-5 NEW (15 etas)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from thesis_render_utils import (
    load_sweep, regime_label, asset_path, save_figure,
    setup_mpl, COLORS
)

setup_mpl()


def main():
    canon = load_sweep('sweep_w58_canonical_full_grid_raw.csv',
                       type_map={'eta': float, 'fund_levy_rate': float,
                                 'seed': int, 'total_bk': int})
    for r in canon:
        r['regime'] = regime_label(r)

    t3 = load_sweep('smoke_eta_sweep_tau3e5_raw.csv',
                    type_map={'eta': float, 'tau': float,
                              'seed': int, 'total_bk': int})
    for r in t3:
        r['regime'] = 'ex_ante_t3'  # NEW canonical candidate

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    series = [
        ('no_tax',     'No tax',                    COLORS['cBlue']),
        ('ex_post',    'Ex-post tax',               COLORS['cRed']),
        ('ex_ante_t5', r'Ex-ante τ=10$^{-5}$ (current canonical)', COLORS['cGreen']),
        ('ex_ante_t3', r'Ex-ante τ=3·10$^{-5}$ (PROPOSED canonical)', COLORS['cBrown']),
        ('ex_ante_t4', r'Ex-ante τ=10$^{-4}$ (default; reference)',  COLORS['cAmber']),
    ]
    for reg, label, color in series:
        if reg == 'ex_ante_t3':
            sub = t3
        else:
            sub = [r for r in canon if r['regime'] == reg]
        if not sub:
            continue
        etas = sorted(set(r['eta'] for r in sub))
        xs, ys, errs = [], [], []
        for eta in etas:
            cell = [r for r in sub if r['eta'] == eta]
            vals = [r['total_bk'] for r in cell]
            xs.append(eta)
            ys.append(np.mean(vals))
            errs.append(np.std(vals))
        if not xs:
            continue
        ys_a, errs_a = np.array(ys), np.array(errs)
        # Highlight the two candidates (t5 and t3) with thicker lines
        lw = 2.0 if reg in ('ex_ante_t5', 'ex_ante_t3') else 1.2
        ax.plot(xs, ys_a, '-o', color=color, label=label, markersize=5, linewidth=lw)
        ax.fill_between(xs, ys_a - errs_a, ys_a + errs_a, alpha=0.15,
                        color=color, linewidth=0)

    ax.axvline(0.10, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlabel(r'Bailout coverage $\eta$')
    ax.set_ylabel('Total bankruptcies (5-seed mean)')
    ax.set_title(r'$\eta$-sweep at canonical w58 cv=0 — '
                 r'$\tau{=}3{\cdot}10^{-5}$ vs $\tau{=}10^{-5}$ vs $\tau{=}10^{-4}$',
                 fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(bottom=2500)

    out = asset_path('ch3_5_claim4_eta', 'tau_compare_eta_sweep.png')
    save_figure(fig, out)
    print(f'\nDONE. Smoking-gun figure: {out}')


if __name__ == '__main__':
    main()
