"""§3.1.x / §3.4 — Per-channel bankruptcy breakdown by fiscal regime at the
locked η-grid {0, 0.10, 0.85} (Phase 2 regen of `3_5_1_regime_channel_breakdown.tex`).

Bold-marker: ex-post η=0.10 (the canonical interior-minimum cell).

Source: sweep_w58_canonical_full_grid_raw.csv (375 sims).
Output: thesis_assets/ch3_5_claim4_eta/3_5_1_regime_channel_breakdown.tex
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from thesis_render_utils import load_sweep, regime_label, asset_path

OUT = asset_path('ch3_5_claim4_eta', '3_5_1_regime_channel_breakdown.tex')

# Locked grid (LD1) + canonical regimes only (LD4)
ETAS = [0.0, 0.10, 0.85]
ETA_TOL = 1e-9
REGIMES = [
    ('no_tax', 'no tax'),
    ('ex_post', 'ex-post'),
    ('ex_ante_t5', 'ex-ante'),
]
CHANNELS = ['total_bk', 'shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths', 'bailout_bill']
BOLD_CELL = ('ex_post', 0.10)


def fmt_int(x):
    """Format integer with thousands separator using LaTeX comma."""
    if x is None:
        return '--'
    return f'{int(round(x)):,}'.replace(',', '{,}')


def main():
    raw = load_sweep('sweep_w58_canonical_full_grid_raw.csv',
                     type_map={'eta': float, 'fund_levy_rate': float, 'seed': int,
                               'total_bk': int, 'shock': int, 'rationing': int,
                               'repay': int, 'contagion': int, 'fiscal_deaths': int,
                               'bailout_bill': float})
    for r in raw:
        r['regime'] = regime_label(r)

    # Aggregate to (regime, eta) -> per-channel mean
    means = {}
    for r in raw:
        key = (r['regime'], r['eta'])
        if r['regime'] not in {x[0] for x in REGIMES}: continue
        if not any(abs(r['eta'] - e) < ETA_TOL for e in ETAS): continue
        means.setdefault(key, []).append(r)

    cell_means = {}
    for key, rows in means.items():
        d = {}
        for ch in CHANNELS:
            d[ch] = float(np.mean([r[ch] for r in rows]))
        cell_means[key] = d

    # Build LaTeX table
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\caption{Per-channel bankruptcy breakdown by fiscal regime at the locked')
    lines.append(r'  $\eta$-grid $\{0, 0.10, 0.85\}$ '
                 r'(5-seed mean, canonical $w58$, $cv = 0$, $\rho = 0.40$, $\alpha = 0.05$).')
    lines.append(r'  The bold row marks ex-post tax at $\eta = 0.10$, the interior-minimum cell.')
    lines.append(r'  At $\eta=0$ all three regimes are mechanically equivalent (no bailout to tax).}')
    lines.append(r'\label{tab:regime_channel_breakdown}')
    lines.append(r'\renewcommand{\arraystretch}{1.05}')
    lines.append(r'\begin{tabular}{rlrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'$\eta$ & Regime & \makecell{total\\bk} & shock & \makecell{ratio-\\ning} & '
                 r'repay & \makecell{conta-\\gion} & \makecell{fiscal\\deaths} & '
                 r'\makecell{bailout\\bill} \\')
    lines.append(r'\midrule')

    for eta in ETAS:
        lines.append(r'\multirow{3}{*}{$' + f'{eta:.2f}'.rstrip('0').rstrip('.') + r'$}')
        for reg_key, reg_label in REGIMES:
            cell = cell_means.get((reg_key, eta), {})
            if not cell:
                row_data = ['--'] * len(CHANNELS)
            else:
                row_data = [fmt_int(cell.get(ch)) for ch in CHANNELS]
            is_bold = (reg_key == BOLD_CELL[0] and abs(eta - BOLD_CELL[1]) < ETA_TOL)
            if is_bold:
                reg_str = r'\textbf{' + reg_label + r'}'
                row_str = ' & '.join(r'\textbf{' + v + r'}' for v in row_data)
            else:
                reg_str = reg_label
                row_str = ' & '.join(row_data)
            lines.append('       & ' + reg_str + ' & ' + row_str + r' \\')
        if eta != ETAS[-1]:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
