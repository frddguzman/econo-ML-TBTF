"""§3.4.3 — ρ-sweep total_bk across 3 regimes (Claim 3 primary figure) — REVISED.

3-panel layout: η ∈ {0, 0.10, 0.85}. Note: at η=0, regimes are equivalent (no bailout),
so all 3 lines coincide visually — we draw all 3 anyway with offset markers so readers
can see the equivalence is intentional.

Sources:
- sweep_w58_rho_raw.csv (nt + ex-post; ex-ante cells at default τ=1e-4 — skip)
- sweep_w58_rho_fund_t5_raw.csv (ex-ante τ=1e-5 canonical, NEW)
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
OUT_5SEED = asset_path('ch3_4_claim3_rho', '3_4_3_rho_sweep_3regimes_5seed.png')

ETAS = [0.0, 0.10, 0.85]


def load_combined():
    rows = []
    for r in load_sweep('sweep_w58_rho_raw.csv', type_map={
        'rho': float, 'eta': float, 'seed': int, 'total_bk': int}):
        if r['fiscal_regime'] == 'none':
            r['regime'] = 'no_tax'
        elif r['fiscal_regime'] == 'socialized_tax':
            r['regime'] = 'ex_post'
        else:
            continue
        rows.append(r)
    for r in load_sweep('sweep_w58_rho_fund_t5_raw.csv', type_map={
        'rho': float, 'eta': float, 'seed': int, 'total_bk': int}):
        r['regime'] = 'ex_ante_t5'
        rows.append(r)
    return rows


def render(rows):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    label_map = {'no_tax': 'No tax', 'ex_post': 'Ex-post tax',
                 'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$'}
    markers = {'no_tax': 'o', 'ex_post': 's', 'ex_ante_t5': '^'}

    for ax, eta in zip(axes, ETAS):
        sub = [r for r in rows if r['eta'] == eta]
        for reg in ['no_tax', 'ex_post', 'ex_ante_t5']:
            cell = [r for r in sub if r['regime'] == reg]
            if not cell: continue
            rhos = sorted(set(r['rho'] for r in cell))
            xs, ys, errs = [], [], []
            for rho in rhos:
                vals = [r['total_bk'] for r in cell if r['rho'] == rho]
                if not vals: continue
                xs.append(rho); ys.append(np.mean(vals)); errs.append(np.std(vals))
            color = REGIME_COLORS.get(reg, 'gray')
            label = label_map[reg]
            # Use distinct markers + slight x-offset to disambiguate at η=0 where regimes overlap
            ax.plot(xs, ys, marker=markers[reg], color=color, label=label,
                    markersize=5, linewidth=1.4, linestyle='-')
            ys_a, errs_a = np.array(ys), np.array(errs)
            ax.fill_between(xs, ys_a - errs_a, ys_a + errs_a, alpha=0.14,
                            color=color, linewidth=0)
        ax.set_title(rf'$\eta = {eta:.2f}$', fontsize=10)
        ax.set_xlabel(r'Fire-sale recovery $\rho$')
        ax.axvline(0.70, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_ylim(bottom=0)
        if ax is axes[0]:
            ax.set_ylabel('Total bankruptcies (5-seed mean)')
            ax.legend(loc='best', fontsize=8)

    fig.suptitle(r'Total bankruptcies vs $\rho$ across 3 regimes (5-seed mean $\pm$ std). '
                 r'Note: at $\eta=0$ regimes are equivalent (no bailout active).',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, OUT_5SEED)


def main():
    rows = load_combined()
    render(rows)


if __name__ == '__main__':
    main()
