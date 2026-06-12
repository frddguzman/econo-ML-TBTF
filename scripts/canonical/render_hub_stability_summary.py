"""§3.1.5 Hub stability summary table (Prof Request C).

5-seed mean ± std for avg_ten / max_ten / avg_cli / turnovers per (regime, η)
at canonical w58 cv=0. From sweep_w58_canonical_full_grid_raw.csv (375 sims).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_render_utils import (
    load_sweep, regime_label, aggregate_by, write_booktabs_table, fmt_msd, asset_path
)

OUT = asset_path('ch3_1_topology', '3_1_5_hub_stability_summary.tex')

REGIMES_ORDER = ['no_tax', 'ex_post', 'ex_ante_t4', 'ex_ante_t5', 'ex_ante_t6']
REGIME_NICE = {
    'no_tax':      'No tax',
    'ex_post':     'Ex-post tax',
    'ex_ante_t4':  r'Ex-ante $\tau{=}10^{-4}$',
    'ex_ante_t5':  r'Ex-ante $\tau{=}10^{-5}$',
    'ex_ante_t6':  r'Ex-ante $\tau{=}10^{-6}$',
}
ETAS = [0.0, 0.10, 0.50, 0.85]


def main():
    rows_raw = load_sweep('sweep_w58_canonical_full_grid_raw.csv',
                          type_map={'eta': float, 'fund_levy_rate': float,
                                    'avg_ten': float, 'max_ten': float,
                                    'avg_cli': float, 'turnovers': float})
    for r in rows_raw:
        r['regime'] = regime_label(r)

    agg = aggregate_by(rows_raw, ('regime', 'eta'),
                       ('avg_ten', 'max_ten', 'avg_cli', 'turnovers'))

    # Build table rows: one row per (regime); sub-cols per (eta, metric)
    headers = ['Regime']
    for eta in ETAS:
        for m in ['avg_ten', 'max_ten', 'avg_cli', 'turnovers']:
            short = {'avg_ten': r'$\overline{\text{ten}}$',
                     'max_ten': r'$\max{\text{ten}}$',
                     'avg_cli': r'$\overline{\text{cli}}$',
                     'turnovers': 'turn'}[m]
            headers.append(rf'$\eta{{=}}{eta:.2f}$ {short}')

    body = []
    for reg in REGIMES_ORDER:
        cells = [REGIME_NICE[reg]]
        for eta in ETAS:
            d = agg.get((reg, eta), {})
            for m in ['avg_ten', 'max_ten', 'avg_cli', 'turnovers']:
                mn, sd, n = d.get(m, (None, None, 0))
                if mn is None:
                    cells.append('--')
                else:
                    decimals = 2 if m == 'avg_ten' else (1 if m == 'avg_cli' else 0)
                    cells.append(fmt_msd(mn, sd, decimals=decimals))
        body.append(cells)

    # Column format: 'l' + 16 'r' (4 etas × 4 metrics)
    col_format = 'l' + 'r' * (len(ETAS) * 4)

    write_booktabs_table(
        rows=body,
        columns=list(range(len(headers))),
        col_headers=headers,
        path=OUT,
        column_format=col_format,
        caption=(r'Hub stability summary at canonical w58 cv=0, 5-seed mean $\pm$ std. '
                 r'avg\_ten = mean tenure of best-lender runs; max\_ten = longest run; '
                 r'avg\_cli = mean clients of best-lender; turn = run count. '
                 r'avg\_ten is bailout-invariant at the mean level; '
                 r'\S 3.2 (Claim 1) uses long-run-count thresholds to reveal regime structure.'),
        label='tab:hub_stability_summary'
    )

    print(f'\nDONE. Table written: {OUT}')


if __name__ == '__main__':
    main()
