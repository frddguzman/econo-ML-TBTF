"""§3.2 — Contagion vs ρ at canonical η=0.10 (single-panel; ρ-sweep chapter).

Per Phase 2 restructure (audit LD3): §3.2 = ρ-sweep at canonical (η=0.10, α=0.05),
3 regimes, single η. This figure replaces the prior 3-panel multi-η variant.

Canonical 3 regimes only (no_tax, ex-post, ex-ante τ=1e-5 from fund_t5 sweep).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, asset_path, save_figure,
    setup_mpl, REGIME_COLORS
)

setup_mpl()
OUT = asset_path('ch3_3_claim2_contagion', '3_3_3_contagion_vs_rho_5seed.png')

ETA_TARGET = 0.10


def load_combined():
    rows = []
    for r in load_sweep('sweep_w58_rho_raw.csv', type_map={
        'rho': float, 'eta': float, 'seed': int, 'contagion': int}):
        if r['fiscal_regime'] == 'none':
            r['regime'] = 'no_tax'
        elif r['fiscal_regime'] == 'socialized_tax':
            r['regime'] = 'ex_post'
        else:
            continue
        rows.append(r)
    for r in load_sweep('sweep_w58_rho_fund_t5_raw.csv', type_map={
        'rho': float, 'eta': float, 'seed': int, 'contagion': int}):
        r['regime'] = 'ex_ante_t5'
        rows.append(r)
    return rows


def main():
    rows = load_combined()
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))
    label_map = {'no_tax': 'No tax', 'ex_post': 'Ex-post tax',
                 'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$'}

    for reg in ['no_tax', 'ex_post', 'ex_ante_t5']:
        sub = [r for r in rows if r['regime'] == reg and abs(r['eta'] - ETA_TARGET) < 1e-9]
        if not sub: continue
        rhos = sorted(set(r['rho'] for r in sub))
        xs, ys, errs = [], [], []
        for rho in rhos:
            vals = [r['contagion'] for r in sub if r['rho'] == rho]
            if not vals: continue
            xs.append(rho); ys.append(np.mean(vals)); errs.append(np.std(vals))
        color = REGIME_COLORS.get(reg, 'gray')
        ys_a, errs_a = np.array(ys), np.array(errs)
        ax.plot(xs, ys_a, '-o', color=color, label=label_map[reg], markersize=5)
        ax.fill_between(xs, ys_a - errs_a/3, ys_a + errs_a/3, alpha=0.18, color=color, linewidth=0)

    ax.axvline(0.30, color='gray', linestyle=':', linewidth=0.8, alpha=0.6,
               label=r'fire-sale-survivor activation $\rho\approx0.3$')
    ax.set_xlabel(r'$\rho$ (fire-sale recovery rate)')
    ax.set_ylabel('Contagion deaths per run')
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=9)

    fig.suptitle(r'Contagion deaths vs $\rho$ at canonical $\eta=0.10$, 3 regimes '
                 r'(5-seed mean $\pm$ std, shaded band)',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, OUT)


if __name__ == '__main__':
    main()
