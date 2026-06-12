"""§3.3.5 — Contagion vs α (cleanest mechanism evidence).

Outputs:
- 3_3_5_contagion_vs_alpha_5seed.png
- 3_3_5_contagion_vs_alpha_s74.png
Source: sweep_w58_alpha_rho_raw.csv (nt+ex-post) + sweep_w58_alpha_rho_fund_t5_raw.csv.

At canonical (ρ=0.40, η=0.10), contagion grows monotonically 76 → 488 across α ∈ [0.02, 0.20].
Algebraic mechanism: collateral term p_j(1-b_j)·α·A_j in eq.6 numerator scales linearly.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, asset_path, save_figure,
    setup_mpl, REGIME_COLORS, SEED_VIS
)

setup_mpl()
OUT_5SEED = asset_path('ch3_3_claim2_contagion', '3_3_5_contagion_vs_alpha_5seed.png')
OUT_S74 = asset_path('ch3_3_claim2_contagion', '3_3_5_contagion_vs_alpha_s74.png')

RHO_FIX = 0.40
ETA_FIX = 0.10


def load_combined():
    rows = []
    for r in load_sweep('sweep_w58_alpha_rho_raw.csv', type_map={
        'alpha_collateral': float, 'rho': float, 'eta': float, 'seed': int,
        'contagion': int}):
        if r['fiscal_regime'] == 'none':
            r['regime'] = 'no_tax'
        elif r['fiscal_regime'] == 'socialized_tax':
            r['regime'] = 'ex_post'
        else:
            continue
        rows.append(r)
    for r in load_sweep('sweep_w58_alpha_rho_fund_t5_raw.csv', type_map={
        'alpha_collateral': float, 'rho': float, 'eta': float, 'seed': int,
        'contagion': int}):
        r['regime'] = 'ex_ante_t5'
        rows.append(r)
    return [r for r in rows if r['rho'] == RHO_FIX and r['eta'] == ETA_FIX]


def render(rows, single_seed=False):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for reg in ['no_tax', 'ex_post', 'ex_ante_t5']:
        sub = [r for r in rows if r['regime'] == reg]
        if not sub: continue
        alphas = sorted(set(r['alpha_collateral'] for r in sub))
        xs, ys, errs = [], [], []
        for a in alphas:
            cell = [r for r in sub if r['alpha_collateral'] == a]
            if single_seed:
                v = next((r['contagion'] for r in cell if r['seed'] == SEED_VIS), None)
                if v is None: continue
                xs.append(a); ys.append(v); errs.append(0)
            else:
                vals = [r['contagion'] for r in cell]
                xs.append(a); ys.append(np.mean(vals)); errs.append(np.std(vals))
        if not xs: continue
        color = REGIME_COLORS.get(reg, 'gray')
        label = {'no_tax': 'No tax', 'ex_post': 'Ex-post tax',
                 'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$'}[reg]
        ys_a, errs_a = np.array(ys), np.array(errs)
        if single_seed or not any(e > 0 for e in errs):
            ax.plot(xs, ys_a, '-o', color=color, label=label, markersize=4)
        else:
            ax.plot(xs, ys_a, '-o', color=color, label=label, markersize=4)
            ax.fill_between(xs, ys_a - errs_a/3, ys_a + errs_a/3, alpha=0.18, color=color, linewidth=0)

    ax.set_xlabel(r'$\alpha$ (collateral recovery coefficient)')
    ax.set_ylabel('Contagion deaths per run')
    suffix = '(SEED=26474)' if single_seed else r'(5-seed mean $\pm$ std, shaded band)'
    ax.set_title(rf'Contagion vs $\alpha$ at canonical ($\rho={RHO_FIX}$, $\eta={ETA_FIX}$) {suffix}')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(bottom=0)
    out = OUT_S74 if single_seed else OUT_5SEED
    save_figure(fig, out)


def main():
    rows = load_combined()
    render(rows, single_seed=False)
    # NOTE: s74 variant skipped — sweep data is at pool {26462-26466}


if __name__ == '__main__':
    main()
