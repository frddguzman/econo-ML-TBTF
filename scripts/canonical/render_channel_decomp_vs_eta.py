"""§3.3.6 (and §3.5.4) — Channel decomposition stacked bar across η, canonical 3 regimes.

Outputs:
- 3_3_6_channel_decomposition_3regimes.png   (3-panel stacked-bar)
- 3_3_6_mortality_table.tex                  (Prof Request A: full mortality breakdown)

REVISED: canonical 3 regimes only (off-canonical τ-variants dropped).
Better colors + label positioning + scale.

Source: sweep_w58_canonical_full_grid_raw.csv (375 sims).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, regime_label, aggregate_by, write_booktabs_table, fmt_msd,
    asset_path, save_figure, setup_mpl, REGIME_LABEL_NICE, COLORS
)

setup_mpl()
OUT_FIG = asset_path('ch3_3_claim2_contagion', '3_3_6_channel_decomposition_3regimes.png')
OUT_TBL = asset_path('ch3_3_claim2_contagion', '3_3_6_mortality_table.tex')
OUT_FIG_CONT = asset_path('ch3_3_claim2_contagion', '3_3_6_1_contagion_focus.png')

CHANNELS = ['shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths']
CHANNEL_LABEL = {
    'shock': 'Shock', 'rationing': 'Rationing', 'repay': 'Repay',
    'contagion': 'Contagion', 'fiscal_deaths': 'Fiscal',
}
# Cleaner color sequence: low-saturation distinct hues (no clashing reds)
CHANNEL_COLOR = {
    'shock':         '#3A6B8A',  # steel blue (largest channel)
    'rationing':     '#7AA88E',  # muted green
    'repay':         '#B0AAB4',  # warm gray
    'contagion':     '#8B1A1A',  # deep red (key claim 2 channel)
    'fiscal_deaths': '#7A4800',  # amber
}

# Canonical body regimes only (LD4)
BODY_REGIMES = ['no_tax', 'ex_post', 'ex_ante_t5']
# Locked η-grid {0, 0.10, 0.85} per LD1 (η=0.50 dropped)
ETAS_TO_PLOT = [0.0, 0.10, 0.85]


def render_figure(raw):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharey=True)
    for ax, reg in zip(axes, BODY_REGIMES):
        sub = [r for r in raw if r['regime'] == reg and r['eta'] in ETAS_TO_PLOT]
        agg = aggregate_by(sub, ('eta',), tuple(CHANNELS))
        x = np.arange(len(ETAS_TO_PLOT))
        bottom = np.zeros(len(ETAS_TO_PLOT))
        for ch in CHANNELS:
            vals = [agg.get((eta,), {}).get(ch, (0, 0, 0))[0] or 0 for eta in ETAS_TO_PLOT]
            ax.bar(x, vals, bottom=bottom, color=CHANNEL_COLOR[ch],
                   label=CHANNEL_LABEL[ch] if ax is axes[0] else None,
                   width=0.65, edgecolor='white', linewidth=0.4)
            bottom += np.array(vals)
        ax.set_title(REGIME_LABEL_NICE.get(reg, reg), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{e:.2f}' for e in ETAS_TO_PLOT])
        ax.set_xlabel(r'$\eta$')
        if ax is axes[0]:
            ax.set_ylabel('Bankruptcies per run (5-seed mean)')
    # Legend at bottom (no overlay with bars)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CHANNEL_COLOR[ch]) for ch in CHANNELS]
    fig.legend(handles, [CHANNEL_LABEL[ch] for ch in CHANNELS],
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02),
               fontsize=9, frameon=False)
    fig.suptitle(r'Channel decomposition vs $\eta$ across canonical 3 regimes (5-seed mean)',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, OUT_FIG)


def render_table(raw):
    """Mortality breakdown — Prof Request A. Canonical 3 regimes."""
    REGIME_NICE = {
        'no_tax': 'No tax', 'ex_post': 'Ex-post',
        'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$',
    }
    sub_body = [r for r in raw if r['regime'] in BODY_REGIMES]
    agg = aggregate_by(sub_body, ('regime', 'eta'),
                       ('total_bk', 'shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths'))

    headers = ['Regime', r'$\eta$', 'Total', 'Shock', 'Rationing', 'Repay', 'Contagion', 'Fiscal']
    body = []
    for reg in BODY_REGIMES:
        for eta in ETAS_TO_PLOT:
            d = agg.get((reg, eta), {})
            cells = [REGIME_NICE.get(reg, reg) if eta == ETAS_TO_PLOT[0] else '',
                     f'{eta:.2f}']
            for k in ['total_bk', 'shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths']:
                mn, sd, _ = d.get(k, (None, None, 0))
                cells.append(fmt_msd(mn, sd, decimals=0) if mn is not None else '--')
            body.append(cells)

    write_booktabs_table(
        rows=body, columns=list(range(len(headers))), col_headers=headers,
        path=OUT_TBL, column_format='lc' + 'r' * 6,
        caption=(r'Mortality breakdown by channel at canonical w58 cv=0 across '
                 r'(regime, $\eta$), 5-seed mean $\pm$ std. Canonical regimes only; '
                 r'off-canonical fund rates $\tau = 10^{-4}$ and $\tau = 10^{-6}$ '
                 r'are not shown.'),
        label='tab:mortality_breakdown'
    )


def render_contagion_focus(raw):
    """§3.3.6.1 — supplementary contagion-only figure (absolute + % of total_bk).

    Two panels: left = absolute contagion deaths vs η for 3 regimes;
    right = contagion as % of total_bk vs η for 3 regimes.
    """
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    REGIME_COLORS = {'no_tax': '#1A3F6B', 'ex_post': '#8B1A1A',
                     'ex_ante_t5': '#1A5C2A'}
    REG_NICE = {'no_tax': 'No tax', 'ex_post': 'Ex-post tax',
                'ex_ante_t5': r'Ex-ante $\tau{=}10^{-5}$'}

    for reg in BODY_REGIMES:
        sub = [r for r in raw if r['regime'] == reg]
        if not sub: continue
        etas = sorted(set(r['eta'] for r in sub))
        agg = aggregate_by(sub, ('eta',), ('contagion', 'total_bk'))
        xs = []
        ys_abs, errs_abs = [], []
        ys_pct, errs_pct = [], []
        for eta in etas:
            d = agg.get((eta,), {})
            cm, cs, _ = d.get('contagion', (None, None, 0))
            tm, ts, _ = d.get('total_bk', (None, None, 0))
            if cm is None or tm is None or tm == 0:
                continue
            xs.append(eta)
            ys_abs.append(cm); errs_abs.append(cs or 0)
            pct = 100 * cm / tm
            # std propagation rough (ignore total_bk variance for simplicity)
            pct_err = 100 * (cs or 0) / tm if tm > 0 else 0
            ys_pct.append(pct); errs_pct.append(pct_err)
        if not xs: continue
        c = REGIME_COLORS[reg]
        label = REG_NICE[reg]
        # Left: absolute
        axes[0].plot(xs, ys_abs, '-o', color=c, label=label, markersize=5, linewidth=1.5)
        ya = np.array(ys_abs); ea = np.array(errs_abs)
        axes[0].fill_between(xs, ya - ea, ya + ea, alpha=0.18, color=c, linewidth=0)
        # Right: %
        axes[1].plot(xs, ys_pct, '-o', color=c, label=label, markersize=5, linewidth=1.5)
        yp = np.array(ys_pct); ep = np.array(errs_pct)
        axes[1].fill_between(xs, yp - ep, yp + ep, alpha=0.18, color=c, linewidth=0)

    axes[0].set_xlabel(r'$\eta$')
    axes[0].set_ylabel('Contagion deaths per run')
    axes[0].set_title('Contagion deaths (absolute)', fontsize=10)
    axes[0].legend(loc='best', fontsize=8)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel(r'$\eta$')
    axes[1].set_ylabel('Contagion as % of total bankruptcies')
    axes[1].set_title('Contagion share of total mortality', fontsize=10)
    axes[1].set_ylim(bottom=0)

    fig.suptitle(r'Contagion-channel focus across 3 regimes (5-seed mean $\pm$ std)',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, OUT_FIG_CONT)


def main():
    raw = load_sweep('sweep_w58_canonical_full_grid_raw.csv',
                     type_map={'eta': float, 'fund_levy_rate': float,
                               'seed': int,
                               'total_bk': int, 'shock': int, 'rationing': int,
                               'repay': int, 'contagion': int, 'fiscal_deaths': int})
    for r in raw:
        r['regime'] = regime_label(r)

    render_figure(raw)
    render_table(raw)
    render_contagion_focus(raw)


if __name__ == '__main__':
    main()
