"""§3.1.x — Hub-stability threshold-tenure table + figure (Phase 2 regen).

Per audit asset table §3 L154: regen as 3 regimes × 3 etas {0, 0.10, 0.85},
cols: regime, η, >30, >50, >100, avg_cli, turnovers. Bold-marker logic:
the ex-post η=0.10 cell (canonical claim-3 co-optimum).

Outputs:
- 3_2_2_threshold_tenure_table.tex   (5-seed mean ± std, 3 regimes × 3 etas)
- 3_2_2_n_runs_30_vs_eta_5seed.png   (3-marker plot, 3 regimes)

Sources:
- sweep_w58_tenure_dist_raw.csv (100 sims, n_runs_gt_* columns)
- sweep_w58_canonical_full_grid_raw.csv (375 sims, avg_cli + turnovers columns)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, aggregate_by, write_booktabs_table, fmt_msd, asset_path,
    regime_label, setup_mpl, REGIME_COLORS, REGIME_LABEL_NICE, SEED_VIS
)

setup_mpl()

OUT_TBL = asset_path('ch3_2_claim1_hub_stability', '3_2_2_threshold_tenure_table.tex')
OUT_FIG_5SEED = asset_path('ch3_2_claim1_hub_stability', '3_2_2_n_runs_30_vs_eta_5seed.png')

# Per LD4: 3 regimes only in body (rf_t1e-4, rf_t1e-6 DROPPED)
REGIME_MAP = {
    'no_tax':    'no_tax',
    'soc_tax':   'ex_post',
    'rf_t1e-5':  'ex_ante_t5',
}
REGIMES_ORDER = ['no_tax', 'ex_post', 'ex_ante_t5']
REGIMES_FIGURE = REGIMES_ORDER  # same 3 in figure
# Per LD1: locked η-grid {0, 0.10, 0.85}
ETAS = [0.0, 0.10, 0.85]
ETA_TOL = 1e-9
# Per audit: keep only >30 (drop >5/>10/>15/>20/>50/>100 per tutor)
THRESHOLDS = [30]
# Bold-marker logic: bold STRIPPED per tutor flag (was ('ex_post', 0.10))
BOLD_CELL = None


def load():
    """Load tenure thresholds + merge in avg_cli/turnovers from canonical full grid."""
    raw_tenure = load_sweep('sweep_w58_tenure_dist_raw.csv', type_map={
        'eta': float, 'seed': int,
        **{f'n_runs_gt_{t}': float for t in THRESHOLDS},
    })
    for r in raw_tenure:
        r['regime'] = REGIME_MAP.get(r['regime_label'], r['regime_label'])

    raw_grid = load_sweep('sweep_w58_canonical_full_grid_raw.csv', type_map={
        'eta': float, 'fund_levy_rate': float, 'seed': int,
        'avg_cli': float, 'turnovers': int,
    })
    for r in raw_grid:
        r['regime'] = regime_label(r)
    return raw_tenure, raw_grid


def render_table(raw_tenure, raw_grid):
    agg_t = aggregate_by(raw_tenure, ('regime', 'eta'),
                         tuple(f'n_runs_gt_{t}' for t in THRESHOLDS))
    agg_g = aggregate_by(raw_grid, ('regime', 'eta'), ('avg_cli', 'turnovers'))

    headers = ['Regime', r'$\eta$'] + [f'>{t}' for t in THRESHOLDS] + ['avg cli', 'turnovers']
    body = []
    nice_label = {'no_tax': 'No tax', 'ex_post': 'Ex-post',
                  'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$'}

    def fmt_bold(s):
        return r'\textbf{' + s + r'}'

    for reg in REGIMES_ORDER:
        for eta in ETAS:
            # Bold disabled per tutor flag — BOLD_CELL is None
            bold = False
            cells = [nice_label[reg] if eta == ETAS[0] else '', f'{eta:.2f}']
            # n_runs > t (from tenure CSV) — std/3 per tutor
            for t in THRESHOLDS:
                mn, sd, _ = agg_t.get((reg, eta), {}).get(f'n_runs_gt_{t}', (None, None, 0))
                if mn is None:
                    s = '--'
                else:
                    s = fmt_msd(mn, sd/3 if sd is not None else None, decimals=1)
                cells.append(fmt_bold(s) if bold else s)
            # avg_cli + turnovers (from canonical full grid) — std/3 per tutor
            for col in ('avg_cli', 'turnovers'):
                mn, sd, _ = agg_g.get((reg, eta), {}).get(col, (None, None, 0))
                if mn is None:
                    s = '--'
                elif col == 'avg_cli':
                    s = fmt_msd(mn, sd/3 if sd is not None else None, decimals=2)
                else:  # turnovers — integer
                    s = fmt_msd(mn, sd/3 if sd is not None else None, decimals=0)
                cells.append(fmt_bold(s) if bold else s)
            body.append(cells)

    write_booktabs_table(
        rows=body,
        columns=list(range(len(headers))),
        col_headers=headers,
        path=OUT_TBL,
        column_format='lc' + 'r' * (len(THRESHOLDS) + 2),
        caption=(r'Hub-stability at canonical w58 cv=0, $\eta$-grid '
                 r'$\{0, 0.10, 0.85\}$. Column $>30$ counts best-lender runs '
                 r'exceeding $30$ periods within a $1000$-period simulation, '
                 r'averaged across $5$ seeds (mean $\pm$ std). '
                 r'\textit{avg cli} is the mean per-period best-lender client count; '
                 r'\textit{turnovers} is the run count minus one '
                 r'(both from canonical full grid, $5$-seed mean $\pm$ std).'),
        label='tab:threshold_tenure'
    )


def render_figure(raw_tenure):
    """Plot n_runs > 30 vs η for each canonical regime, 3-marker grid."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for reg in REGIMES_FIGURE:
        xs, ys, errs = [], [], []
        for eta in ETAS:
            cell = [r for r in raw_tenure if r['regime'] == reg
                    and abs(float(r['eta']) - eta) < ETA_TOL]
            if not cell: continue
            vals = [r['n_runs_gt_30'] for r in cell]
            xs.append(eta); ys.append(np.mean(vals)); errs.append(np.std(vals))
        if not xs: continue
        color = REGIME_COLORS.get(reg, 'gray')
        label = REGIME_LABEL_NICE.get(reg, reg)
        ys = np.array(ys); errs = np.array(errs)
        ax.plot(xs, ys, '-o', color=color, label=label, markersize=6, linewidth=1.6)
        ax.fill_between(xs, ys-errs/3, ys+errs/3, alpha=0.18, color=color, linewidth=0)

    ax.set_xlabel(r'Bailout coverage $\eta$')
    ax.set_ylabel(r'$n_{\text{runs}} > 30$ periods')
    ax.set_title(r'Long-tenure hub runs vs $\eta$ at canonical w58 cv=0 '
                 r'(5-seed mean $\pm$ std)')
    ax.legend(loc='best', fontsize=9)
    ax.set_xticks(ETAS)
    ax.set_ylim(bottom=0)

    from thesis_render_utils import save_figure
    save_figure(fig, OUT_FIG_5SEED)


def main():
    raw_tenure, raw_grid = load()
    render_table(raw_tenure, raw_grid)
    render_figure(raw_tenure)
    print(f'\nDONE. Outputs: {OUT_TBL}, {OUT_FIG_5SEED}')


if __name__ == '__main__':
    main()
