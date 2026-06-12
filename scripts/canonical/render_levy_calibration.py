"""§3.5.6 — Levy calibration: why τ=1e-5 not τ=1e-4 — REVISED.

Outputs:
- 3_5_6_levy_calibration_table.tex
- 3_5_6_levy_calibration.png

Per USER 2026-05-09: drop contagion line; only total_bk + fiscal deaths needed.

Sources:
- sweep_w58_tau_raw.csv     (300 rows; τ ∈ {5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3})
- sweep_w58_tau_low_raw.csv (200 rows; τ ∈ {0, 1e-6, 1e-5, 5e-5})
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from thesis_render_utils import (
    load_sweep, aggregate_by, write_booktabs_table, fmt_msd,
    asset_path, save_figure, setup_mpl, COLORS
)

setup_mpl()
OUT_TBL = asset_path('ch3_5_claim4_eta', '3_5_6_levy_calibration_table.tex')
OUT_FIG = asset_path('ch3_5_claim4_eta', '3_5_6_levy_calibration.png')


def main():
    rows = []
    for f in ['sweep_w58_tau_raw.csv', 'sweep_w58_tau_low_raw.csv']:
        rows.extend(load_sweep(f, type_map={
            'tau': float, 'eta': float, 'seed': int,
            'total_bk': int, 'contagion': int, 'fiscal_deaths': int,
            'shock': int, 'rationing': int, 'repay': int}))
    print(f'Loaded {len(rows)} tau-sweep rows')

    taus = sorted(set(r['tau'] for r in rows))
    etas = sorted(set(r['eta'] for r in rows))

    agg = aggregate_by(rows, ('tau', 'eta'),
                       ('total_bk', 'contagion', 'fiscal_deaths', 'rationing'))

    eta_focus = 0.10
    if eta_focus not in etas:
        eta_focus = etas[1] if len(etas) > 1 else etas[0]

    headers = [r'$\tau$ (levy rate)', 'Total bk', 'Fiscal deaths', 'Rationing']
    body = []
    for tau in taus:
        d = agg.get((tau, eta_focus), {})
        cells = [f'{tau:.0e}']
        for k in ['total_bk', 'fiscal_deaths', 'rationing']:
            mn, sd, _ = d.get(k, (None, None, 0))
            cells.append(fmt_msd(mn, sd, decimals=0) if mn is not None else '--')
        body.append(cells)

    write_booktabs_table(
        rows=body, columns=list(range(len(headers))), col_headers=headers,
        path=OUT_TBL, column_format='lrrr',
        caption=(rf'Levy calibration sensitivity at canonical w58 cv=0, '
                 rf'$\eta = {eta_focus:.2f}$, ex-ante regime, 5-seed mean $\pm$ std. '
                 rf'High $\tau$ raises total bankruptcies via fiscal-deaths channel '
                 rf'irrespective of TBTF outcomes; canonical $\tau = 10^{{-5}}$ minimises '
                 rf'fiscal drag while preserving Claim 4 interior optimum.'),
        label='tab:levy_calibration'
    )

    # Figure: total_bk + fiscal_deaths vs tau (log scale x); contagion dropped per USER
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    means_total, means_fiscal = [], []
    for tau in taus:
        d = agg.get((tau, eta_focus), {})
        means_total.append(d.get('total_bk', (None,))[0] or 0)
        means_fiscal.append(d.get('fiscal_deaths', (None,))[0] or 0)

    ax.plot(taus, means_total, '-o', color=COLORS['cBlue'], label='Total bankruptcies', markersize=6, linewidth=1.5)
    ax.plot(taus, means_fiscal, '-s', color=COLORS['cAmber'], label='Fiscal deaths', markersize=5, linewidth=1.5)
    ax.set_xscale('log')
    ax.axvline(1e-5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.annotate(r'canonical $\tau=10^{-5}$',
                xy=(1e-5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1000),
                xytext=(1.5e-5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 1000),
                fontsize=8, color='gray', va='center')
    ax.set_xlabel(r'$\tau$ (resolution-fund levy rate, log scale)')
    ax.set_ylabel(rf'Events per run at $\eta = {eta_focus}$ (5-seed mean)')
    ax.set_title(r'Levy calibration: high $\tau$ kills via levy alone')
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(bottom=0)
    save_figure(fig, OUT_FIG)


if __name__ == '__main__':
    main()
