"""§3.1.3 — DDF of in-degree from topology JSONs (REVISED).

bank_detail.clients column is binary/wrong (max=1). Use topology JSON edges to compute
true in-degree per lender per snapshot. Pool across the 3 snapshots per regime.

Outputs:
- 3_1_3_ddf_indegree.png
- 3_1_3_ddf_powerlaw_fit.tex
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import (
    asset_path, save_figure, setup_mpl, COLORS, write_booktabs_table
)

setup_mpl()
OUT_FIG = asset_path('ch3_1_topology', '3_1_3_ddf_indegree.png')
OUT_TBL = asset_path('ch3_1_topology', '3_1_3_ddf_powerlaw_fit.tex')

REGIMES = [
    ('nt',    'No tax',                      '../Simulations/topology_w58_nt_e01.json',  COLORS['cBlue']),
    ('st',    'Ex-post tax',                 '../Simulations/topology_w58_st_e01.json',  COLORS['cRed']),
    ('t5_rf', r'Ex-ante $\tau{=}10^{-5}$',   '../Simulations/topology_w58_t5_rf_e01.json', COLORS['cGreen']),
]


def in_degrees_from_topology(json_path):
    """Pool in-degree counts across the 3 snapshots in a topology JSON."""
    if not os.path.exists(json_path):
        return np.array([])
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    pooled = []
    for k, snap in d.items():
        edges = snap.get('edges', [])
        # Count edges per lender_id
        deg = {}
        for e in edges:
            try:
                l = int(e['lender'])
                deg[l] = deg.get(l, 0) + 1
            except (ValueError, KeyError, TypeError):
                continue
        # Include all banks (50) — banks with 0 in-degree are also part of dist
        n_banks = 50
        for bank_id in range(n_banks):
            pooled.append(deg.get(bank_id, 0))
    return np.array(pooled)


def ddf(values):
    values = values[values > 0]
    if len(values) == 0:
        return np.array([]), np.array([])
    ks = np.arange(1, values.max() + 1)
    p = np.array([(values >= k).mean() for k in ks])
    return ks, p


def mlm_powerlaw_exponent(values, kmin=2):
    x = values[values >= kmin]
    if len(x) < 5:
        return None, None
    n = len(x)
    gamma = 1 + n / np.sum(np.log(x / (kmin - 0.5)))
    se = (gamma - 1) / np.sqrt(n)
    return gamma, se


def main():
    # First pass: collect all in-degree distributions to get global x-range
    all_distributions = []
    for reg_tag, label, path, color in REGIMES:
        in_deg = in_degrees_from_topology(path)
        all_distributions.append((label, color, in_deg))
    # Global k_max for unified x-axis
    global_k_max = max((d.max() if len(d) else 0) for _, _, d in all_distributions)
    if global_k_max < 2: global_k_max = 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    table_rows = []
    for ax, (label, color, in_deg) in zip(axes, all_distributions):
        if len(in_deg) == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=9)
            table_rows.append([label, '--', '--', '--'])
            ax.set_title(label, fontsize=10)
            continue

        ks, p = ddf(in_deg)
        if len(ks) > 0:
            ax.loglog(ks, p, '-o', color=color, markersize=4, linewidth=1.2)
        ax.set_xlim(0.9, global_k_max * 1.2)
        ax.set_ylim(1e-3, 1.2)
        ax.set_xlabel('In-degree $k$')
        if ax is axes[0]:
            ax.set_ylabel(r'$P(K \geq k)$')
        ax.set_title(label, fontsize=10)
        # Stats
        gamma, se = mlm_powerlaw_exponent(in_deg, kmin=2)
        n_obs = int(np.sum(in_deg > 0))
        k_max = int(in_deg.max()) if len(in_deg) > 0 else 0
        if gamma is not None:
            table_rows.append([label, f'{n_obs}',
                               f'{gamma:.2f} $\\pm$ {se:.2f}', f'{k_max}'])
        else:
            table_rows.append([label, f'{n_obs}', '--', f'{k_max}'])

    fig.suptitle('Decumulative degree distribution (in-degree) at canonical w58 cv=0, $\\eta=0.10$ '
                 '— pooled across 3 snapshots per regime', fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, OUT_FIG)

    write_booktabs_table(
        rows=table_rows,
        columns=[0, 1, 2, 3],
        col_headers=['Regime', '$N$ obs', r'$\hat\gamma$ (MLM)', r'$k_{\max}$'],
        path=OUT_TBL,
        column_format='lrrr',
        caption=(r'In-degree power-law tail exponent at canonical w58 cv=0, $\eta=0.10$, '
                 r'pooled across 3 topology snapshots per regime. MLM estimate '
                 r'(Clauset-style for discrete distributions, $k_{\min}=2$). '
                 r'Lineage anchor: Lenzu-Tedeschi 2012 reports $\gamma_{DDF} \in [2,3]$ for '
                 r'star-like configurations; Brini-Tedeschi-Tantari 2023 Fig.4 right.'),
        label='tab:ddf_powerlaw_fit'
    )


if __name__ == '__main__':
    main()
