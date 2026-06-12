"""§3.1.x — Contagion deaths vs η at 3-marker grid {0, 0.10, 0.85}.

Per Phase 2 restructure (audit LD1): η-grid locked to {0, 0.10, 0.85}.
The figure shows the three grid anchors as marker points; intermediate η values
filtered out for grid consistency across §3.1 figures.

Outputs:
- 3_3_4_contagion_vs_eta_5seed.png
Source: sweep_w58_canonical_full_grid_raw.csv (375 sims, filtered to 3 etas).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, regime_label, asset_path, save_figure,
    setup_mpl, REGIME_COLORS, REGIME_LABEL_NICE, SEED_VIS
)

setup_mpl()
OUT_5SEED = asset_path('ch3_3_claim2_contagion', '3_3_4_contagion_vs_eta_5seed.png')
OUT_S74 = asset_path('ch3_3_claim2_contagion', '3_3_4_contagion_vs_eta_s74.png')

# Locked η-grid for §3.1 figures (LD1)
ETA_GRID = [0.0, 0.10, 0.85]
ETA_TOL = 1e-9


def render(raw, single_seed=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    # Body figures: canonical 3 regimes only (τ-variants live in §3.5.6 levy section)
    for reg in ['no_tax', 'ex_post', 'ex_ante_t5']:
        sub = [r for r in raw if r['regime'] == reg]
        if not sub: continue
        # Filter to locked η-grid {0, 0.10, 0.85} per LD1
        etas = sorted(e for e in ETA_GRID
                      if any(abs(r['eta'] - e) < ETA_TOL for r in sub))
        xs, ys, errs = [], [], []
        for eta in etas:
            cell = [r for r in sub if abs(r['eta'] - eta) < ETA_TOL]
            if single_seed:
                v = next((r['contagion'] for r in cell if r['seed'] == SEED_VIS), None)
                if v is None: continue
                xs.append(eta); ys.append(v); errs.append(0)
            else:
                vals = [r['contagion'] for r in cell]
                xs.append(eta); ys.append(np.mean(vals)); errs.append(np.std(vals))
        if not xs: continue
        color = REGIME_COLORS.get(reg, 'gray')
        label = REGIME_LABEL_NICE.get(reg, reg)
        ys_a, errs_a = np.array(ys), np.array(errs)
        if single_seed or not any(e > 0 for e in errs):
            ax.plot(xs, ys_a, '-o', color=color, label=label, markersize=4)
        else:
            ax.plot(xs, ys_a, '-o', color=color, label=label, markersize=4)
            ax.fill_between(xs, ys_a - errs_a/3, ys_a + errs_a/3, alpha=0.18, color=color, linewidth=0)

    ax.set_xlabel(r'Bailout coverage $\eta$')
    ax.set_ylabel('Contagion deaths per run')
    suffix = '(SEED=26474)' if single_seed else r'(5-seed mean $\pm$ std, shaded band)'
    ax.set_title(f'Contagion deaths vs $\\eta$ across regimes {suffix}')
    ax.set_xticks(ETA_GRID)
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(bottom=0)
    out = OUT_S74 if single_seed else OUT_5SEED
    save_figure(fig, out)


def main():
    raw = load_sweep('sweep_w58_canonical_full_grid_raw.csv',
                     type_map={'eta': float, 'fund_levy_rate': float,
                               'seed': int, 'contagion': int})
    for r in raw:
        r['regime'] = regime_label(r)
    render(raw, single_seed=False)
    # NOTE: s74 variant skipped — sweep data is at pool {26462-26466}


if __name__ == '__main__':
    main()
