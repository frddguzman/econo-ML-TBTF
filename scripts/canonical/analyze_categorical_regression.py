"""§3.5.7 — Categorical regression of system variables on η (Brini Tables 1-2 style).

Output: 3_5_7_categorical_regression.tex (booktabs LaTeX table)

For each outcome (total_bk, contagion, fiscal_deaths, rationing), regress on:
- η (continuous)
- Regime dummies (no_tax baseline; ex_post and ex_ante_t5 indicators)
- η × regime interactions

Uses statsmodels OLS. 5-seed pool aggregates (375 obs from canonical full grid).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_render_utils import (
    load_sweep, regime_label, write_booktabs_table, asset_path
)

OUT = asset_path('ch3_5_claim4_eta', '3_5_7_categorical_regression.tex')


def main():
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        import pandas as pd
    except ImportError as e:
        print(f'  [skip] statsmodels/pandas not available: {e}')
        return

    raw = load_sweep('sweep_w58_canonical_full_grid_raw.csv', type_map={
        'eta': float, 'fund_levy_rate': float, 'seed': int,
        'total_bk': int, 'contagion': int, 'fiscal_deaths': int,
        'rationing': int, 'shock': int})
    for r in raw:
        r['regime'] = regime_label(r)

    # Filter to body regimes only (no_tax, ex_post, ex_ante_t5)
    body = [r for r in raw if r['regime'] in ('no_tax', 'ex_post', 'ex_ante_t5')]
    df = pd.DataFrame(body)
    df['ex_post'] = (df['regime'] == 'ex_post').astype(int)
    df['ex_ante_t5'] = (df['regime'] == 'ex_ante_t5').astype(int)
    df['eta_x_ex_post'] = df['eta'] * df['ex_post']
    df['eta_x_ex_ante_t5'] = df['eta'] * df['ex_ante_t5']

    outcomes = [
        ('total_bk', 'Total bankruptcies'),
        ('contagion', 'Contagion deaths'),
        ('fiscal_deaths', 'Fiscal deaths'),
        ('rationing', 'Rationing deaths'),
    ]

    headers = ['Predictor'] + [label for _, label in outcomes]
    predictors = ['Intercept', 'eta', 'ex_post', 'ex_ante_t5',
                  'eta_x_ex_post', 'eta_x_ex_ante_t5']
    pred_labels = ['Intercept', r'$\eta$', 'Ex-post', r'Ex-ante $\tau{=}10^{-5}$',
                   r'$\eta \times$ Ex-post', r'$\eta \times$ Ex-ante']

    # Run OLS for each outcome, collect coefficients
    coef_table = {p: [] for p in predictors}
    se_table = {p: [] for p in predictors}
    rsq = []
    nn = []
    for out_var, _ in outcomes:
        formula = (f'{out_var} ~ eta + ex_post + ex_ante_t5 '
                   f'+ eta_x_ex_post + eta_x_ex_ante_t5')
        model = smf.ols(formula=formula, data=df).fit()
        rsq.append(model.rsquared)
        nn.append(int(model.nobs))
        for p in predictors:
            beta = model.params.get(p, None)
            se = model.bse.get(p, None)
            pval = model.pvalues.get(p, 1.0)
            stars = '$^{***}$' if pval < 0.01 else ('$^{**}$' if pval < 0.05
                    else ('$^{*}$' if pval < 0.10 else ''))
            coef_str = f'{beta:.1f}{stars}' if beta is not None else '--'
            se_str = f'({se:.1f})' if se is not None else ''
            coef_table[p].append(coef_str)
            se_table[p].append(se_str)

    # Build table body
    body_rows = []
    for p, label in zip(predictors, pred_labels):
        body_rows.append([label] + coef_table[p])
        body_rows.append([''] + se_table[p])
    body_rows.append([r'\midrule $R^2$'] + [f'{r:.3f}' for r in rsq])
    body_rows.append(['$N$'] + [str(n) for n in nn])

    write_booktabs_table(
        rows=body_rows, columns=list(range(len(headers))), col_headers=headers,
        path=OUT, column_format='l' + 'r' * len(outcomes),
        caption=(r'OLS coefficients for system outcomes regressed on $\eta$ '
                 r'with regime dummies and $\eta \times$ regime interactions '
                 r'(no-tax = baseline). Standard errors in parentheses. '
                 r'Significance: $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. '
                 r'Source: \texttt{sweep\_w58\_canonical\_full\_grid\_raw.csv} (5-seed pool, '
                 r'body regimes only). Brini-Tedeschi-Tantari 2023 Tables 1-2 analogue.'),
        label='tab:categorical_regression'
    )
    print(f'\nDONE. Table written: {OUT}')


if __name__ == '__main__':
    main()
