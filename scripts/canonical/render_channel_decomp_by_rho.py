"""§3.4.5 — Channel decomposition by ρ (Claim 3 mechanism) — CONVERTED TO TABLE.

Per USER 2026-05-09: convert from stacked-bar figure to booktabs table; mixing
fire-sale survivors (counts) and bankruptcies (different scale) on one y-axis is misleading.

Output: 3_4_5_rho_channel_decomposition.tex   (replaces .png)

Source: canonical 3 regimes at η=0.10 across full ρ-sweep.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thesis_render_utils import (
    load_sweep, aggregate_by, write_booktabs_table, fmt_msd, asset_path
)

OUT_TBL = asset_path('ch3_4_claim3_rho', '3_4_5_rho_channel_decomposition.tex')

CHANNELS = ['shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths', 'zombies']
CHANNEL_LABEL = {
    'shock': 'Shock', 'rationing': 'Rationing', 'repay': 'Repay',
    'contagion': 'Contagion', 'fiscal_deaths': 'Fiscal',
    'zombies': 'Fire-sale survivors',
}
ETA_FIX = 0.10


def main():
    rows = []
    for r in load_sweep('sweep_w58_rho_raw.csv', type_map={
        'rho': float, 'eta': float, 'seed': int,
        **{ch: int for ch in CHANNELS}}):
        if r['fiscal_regime'] == 'none':
            r['regime'] = 'no_tax'
        elif r['fiscal_regime'] == 'socialized_tax':
            r['regime'] = 'ex_post'
        else:
            continue
        rows.append(r)
    for r in load_sweep('sweep_w58_rho_fund_t5_raw.csv', type_map={
        'rho': float, 'eta': float, 'seed': int,
        **{ch: int for ch in CHANNELS}}):
        r['regime'] = 'ex_ante_t5'
        rows.append(r)

    sub = [r for r in rows if r['eta'] == ETA_FIX]
    rhos = sorted(set(r['rho'] for r in sub))

    REGIMES_ORDER = ['no_tax', 'ex_post', 'ex_ante_t5']
    REG_NICE = {'no_tax': 'No tax', 'ex_post': 'Ex-post',
                'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$'}

    agg = aggregate_by(sub, ('regime', 'rho'), tuple(CHANNELS))

    headers = ['Regime', r'$\rho$'] + [CHANNEL_LABEL[c] for c in CHANNELS]
    body = []
    for reg in REGIMES_ORDER:
        for rho in rhos:
            d = agg.get((reg, rho), {})
            cells = [REG_NICE.get(reg, reg) if rho == rhos[0] else '',
                     f'{rho:.2f}']
            for ch in CHANNELS:
                mn, sd, _ = d.get(ch, (None, None, 0))
                cells.append(fmt_msd(mn, sd, decimals=0) if mn is not None else '--')
            body.append(cells)

    write_booktabs_table(
        rows=body, columns=list(range(len(headers))), col_headers=headers,
        path=OUT_TBL, column_format='lc' + 'r' * len(CHANNELS),
        caption=(r'Channel decomposition by $\rho$ at canonical w58 cv=0, $\eta=0.10$, '
                 r'5-seed mean $\pm$ std. Bankruptcy channels (shock, rationing, repay, '
                 r'contagion, fiscal) on one count scale; fire-sale survivors are '
                 r'survivor-events (different mechanism, separate column for clarity). '
                 r'Both fire-sale-survivor count and contagion peak at $\rho=0.70$, '
                 r'collapsing at $\rho=0.90$ (survivors heal too fast).'),
        label='tab:rho_channel_decomp'
    )


if __name__ == '__main__':
    main()
