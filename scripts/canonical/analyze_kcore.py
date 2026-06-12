"""§3.1.4 — K-core decomposition (BT 2017 §3.1 analogue) — REVISED.

Drops the loan>0 filter (loans are 0 in JSON post-forward) and uses all edges from topology JSONs.
3 regimes; expects nt + st + t5_rf JSONs in ../Simulations/.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from thesis_render_utils import asset_path, save_figure, setup_mpl, COLORS

setup_mpl()
OUT = asset_path('ch3_1_topology', '3_1_4_kcore.png')

REGIMES = [
    ('nt',    'No tax',                      '../Simulations/topology_w58_nt_e01.json'),
    ('st',    'Ex-post tax',                 '../Simulations/topology_w58_st_e01.json'),
    ('t5_rf', r'Ex-ante $\tau{=}10^{-5}$',   '../Simulations/topology_w58_t5_rf_e01.json'),
]


def kcore_layers(snapshot):
    """Return per-bank k-core number using networkx; drop loan>0 filter."""
    try:
        import networkx as nx
    except ImportError:
        return None
    edges = snapshot.get('edges', [])
    G = nx.Graph()
    for e in edges:
        try:
            b, l = int(e['borrower']), int(e['lender'])
            if b != l:  # no self-loops
                G.add_edge(b, l)
        except (ValueError, KeyError, TypeError):
            continue
    if G.number_of_nodes() == 0:
        return None
    # k_core requires no self-loops; we already filtered
    core = nx.core_number(G)
    return core


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (reg_tag, label, json_path) in zip(axes, REGIMES):
        if json_path is None or not os.path.exists(json_path):
            ax.text(0.5, 0.5, f'NO DATA\n{reg_tag}', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=10)
            ax.set_title(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        with open(json_path, encoding='utf-8') as f:
            d = json.load(f)
        snapshot = d.get('999', {})
        cores = kcore_layers(snapshot)
        if cores is None or not cores:
            ax.text(0.5, 0.5, 'no edges or networkx missing', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=9)
            ax.set_title(label, fontsize=10)
            continue

        vals = list(cores.values())
        max_k = max(vals)
        bins = np.arange(0, max_k + 2) - 0.5
        ax.hist(vals, bins=bins, color=COLORS['cBlue'], edgecolor='white',
                linewidth=0.5)
        ax.set_xlabel('k-core number')
        if ax is axes[0]:
            ax.set_ylabel('Bank count')
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(0, max_k + 1))

        n_in_top = sum(1 for v in vals if v == max_k)
        # Annotate near-bottom-right to avoid overlap with bars at low k
        ax.annotate(f'top k-core: {n_in_top} banks (k={max_k})',
                    xy=(0.97, 0.95), xycoords='axes fraction', fontsize=8,
                    color=COLORS['cGray'], va='top', ha='right')

    fig.suptitle('K-core decomposition at $t=999$, canonical w58 cv=0, $\\eta=0.10$ '
                 '(low max-k expected: hub-and-spoke topology)',
                 fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, OUT)


if __name__ == '__main__':
    main()
