"""§3.2.1 avg_ten decoupling table.

Compact 5-regime × 4-eta table showing avg_ten 5-seed mean ± std.
Sets up §3.2.2 (the threshold-tenure reveal) by showing the surface-level flatness.
Source: sweep_w58_tenure_dist_raw.csv (100 sims).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_render_utils import (
    load_sweep, regime_label, aggregate_by, write_booktabs_table, fmt_msd, asset_path
)

OUT = asset_path('ch3_2_claim1_hub_stability', '3_2_1_avg_ten_decoupling.tex')

# tenure_dist file uses regime_label column directly (no_tax / soc_tax / rf_t1e-X)
REGIME_MAP = {
    'no_tax':    ('no_tax',     'No tax'),
    'soc_tax':   ('ex_post',    'Ex-post tax'),
    'rf_t1e-4':  ('ex_ante_t4', r'Ex-ante $\tau{=}10^{-4}$'),
    'rf_t1e-5':  ('ex_ante_t5', r'Ex-ante $\tau{=}10^{-5}$'),
    'rf_t1e-6':  ('ex_ante_t6', r'Ex-ante $\tau{=}10^{-6}$'),
}
REGIMES_ORDER = ['no_tax', 'ex_post', 'ex_ante_t4', 'ex_ante_t5', 'ex_ante_t6']
ETAS = [0.0, 0.10, 0.50, 0.85]


def main():
    raw = load_sweep('sweep_w58_tenure_dist_raw.csv',
                     type_map={'eta': float, 'avg_ten': float})
    for r in raw:
        nice = REGIME_MAP.get(r['regime_label'])
        if nice is None: continue
        r['regime'] = nice[0]

    agg = aggregate_by([r for r in raw if 'regime' in r],
                       ('regime', 'eta'), ('avg_ten',))

    headers = ['Regime'] + [rf'$\eta{{=}}{e:.2f}$' for e in ETAS]
    body = []
    for reg in REGIMES_ORDER:
        nice = next((REGIME_MAP[k][1] for k in REGIME_MAP if REGIME_MAP[k][0] == reg), reg)
        cells = [nice]
        for eta in ETAS:
            mn, sd, n = agg.get((reg, eta), {}).get('avg_ten', (None, None, 0))
            cells.append(fmt_msd(mn, sd, decimals=2) if mn is not None else '--')
        body.append(cells)

    write_booktabs_table(
        rows=body,
        columns=list(range(len(headers))),
        col_headers=headers,
        path=OUT,
        column_format='l' + 'r' * len(ETAS),
        caption=(r'avg\_ten at canonical w58 cv=0 across (regime, $\eta$), '
                 r'5-seed mean $\pm$ std. The values cluster in $[3.02, 3.50]$ across '
                 r'all 20 cells, suggesting bailout-invariant hub stability at the '
                 r'mean-tenure level. \S 3.2.2 reveals this surface picture is '
                 r'misleading: long-run-count thresholds expose a regime-conditional '
                 r'$\eta$-optimum that avg\_ten averages out.'),
        label='tab:avg_ten_decoupling'
    )

    print(f'\nDONE. Table written: {OUT}')


if __name__ == '__main__':
    main()
