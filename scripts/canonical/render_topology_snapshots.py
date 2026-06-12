"""§3.1.2 — Topology snapshots (Brini Fig.3 analogue) — REVISED.

Spring layout (force-directed) instead of circular; visible edges (drop loan>0 filter
since loan amounts are post-period reset to 0); node size by in-degree.
3 regimes × 4 timesteps = 12 panels.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import csv
import matplotlib.pyplot as plt
from thesis_render_utils import asset_path, save_figure, setup_mpl, COLORS

setup_mpl()
OUT = asset_path('ch3_1_topology', '3_1_2_topology_snapshots.png')

# All 3 regimes available (nt + st + t5_rf JSONs all dumped)
REGIMES = [
    ('nt',    'No tax',                      '../Simulations/topology_w58_nt_e01.json'),
    ('st',    'Ex-post tax',                 '../Simulations/topology_w58_st_e01.json'),
    ('t5_rf', r'Ex-ante $\tau{=}10^{-5}$',   '../Simulations/topology_w58_t5_rf_e01.json'),
]


def get_peak_client_t(regime_tag):
    """Find period with highest best_lender_clients in dashboard CSV."""
    if regime_tag == 't5_rf':
        path = '../Simulations/dash_w58_t5_rf_e01.csv'
    else:
        path = f'../Simulations/dash_w58_{regime_tag}_e01.csv'
    if not os.path.exists(path):
        return 999
    best_t, best_val = 999, -1
    for r in csv.DictReader(open(path, encoding='utf-8')):
        try:
            cli = float(r['best_lender_clients'])
            t = int(float(r['time']))
            if cli > best_val:
                best_val, best_t = cli, t
        except (ValueError, TypeError):
            continue
    return best_t


def spring_layout(n_nodes, edges, iterations=80, seed=42):
    """Simple Fruchterman-Reingold spring layout (no networkx dependency)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=1.0, size=(n_nodes, 2))
    if not edges:
        return pos
    k = 1.0 / np.sqrt(n_nodes)  # ideal edge length
    for it in range(iterations):
        t = 0.1 * (1 - it / iterations)  # cooling
        disp = np.zeros_like(pos)
        # Repulsive
        for i in range(n_nodes):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1) + 1e-6
            force = (k * k) / dist
            disp[i] += np.sum(delta / dist[:, None] * force[:, None], axis=0)
        # Attractive
        for (a, b) in edges:
            delta = pos[a] - pos[b]
            dist = np.linalg.norm(delta) + 1e-6
            force = (dist * dist) / k
            disp[a] -= delta / dist * force
            disp[b] += delta / dist * force
        # Update
        norm = np.linalg.norm(disp, axis=1) + 1e-6
        pos += disp / norm[:, None] * np.minimum(norm, t)[:, None]
    return pos


def render_network(ax, snapshot, title, n_banks=50):
    if snapshot is None:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        return

    edges_raw = snapshot.get('edges', [])
    edge_pairs = []
    for e in edges_raw:
        try:
            b, l = int(e['borrower']), int(e['lender'])
            if 0 <= b < n_banks and 0 <= l < n_banks and b != l:
                edge_pairs.append((b, l))
        except (ValueError, KeyError, TypeError):
            continue

    pos = spring_layout(n_banks, edge_pairs)

    # Node sizes by in-degree (number of banks borrowing from each)
    in_degree = np.zeros(n_banks)
    for (b, l) in edge_pairs:
        in_degree[l] += 1
    max_deg = max(1, in_degree.max())
    sizes = 8 + 80 * (in_degree / max_deg)
    colors = np.where(in_degree > 0, COLORS['cRed'], COLORS['cGray'])
    # Highlight hub
    if in_degree.max() > 0:
        hub = int(np.argmax(in_degree))
        colors = list(colors)
        colors[hub] = COLORS['cBlue']

    # Draw edges
    for (b, l) in edge_pairs:
        x0, y0 = pos[b]; x1, y1 = pos[l]
        ax.plot([x0, x1], [y0, y1], color=COLORS['cGray'], linewidth=0.4, alpha=0.5)

    # Draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=sizes, edgecolors='black',
               linewidths=0.3, zorder=3)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=8)
    # Unified panel extent: pad to a common size so all panels look uniform
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    span = max(pos[:, 0].max() - pos[:, 0].min(),
               pos[:, 1].max() - pos[:, 1].min(), 0.5) * 0.6 + 0.2
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)


def main():
    fig, axes = plt.subplots(len(REGIMES), 4, figsize=(13, 9))
    for i, (reg_tag, reg_label, json_path) in enumerate(REGIMES):
        snapshots = {0: None, 500: None, 999: None}
        if json_path and os.path.exists(json_path):
            with open(json_path, encoding='utf-8') as f:
                d = json.load(f)
            for k in ['0', '500', '999']:
                if k in d:
                    snapshots[int(k)] = d[k]

        peak_t = get_peak_client_t(reg_tag)
        peak_snapshot = snapshots.get(min([0, 500, 999], key=lambda x: abs(x - peak_t)))
        peak_label = f't={peak_t} (peak)'

        cols = [(0, 't=0', snapshots[0]),
                (500, 't=500', snapshots[500]),
                (999, 't=999', snapshots[999]),
                (peak_t, peak_label, peak_snapshot)]
        for j, (t, label, snap) in enumerate(cols):
            ax = axes[i, j]
            render_network(ax, snap, label)
            if j == 0:
                ax.text(-0.08, 0.5, reg_label, transform=ax.transAxes,
                        rotation=90, va='center', ha='right', fontsize=10)

    fig.suptitle('Topology snapshots at canonical w58 cv=0 — spring layout, hub in blue, '
                 'lenders in red, periphery grey', fontsize=10, y=0.99)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    save_figure(fig, OUT)


if __name__ == '__main__':
    main()
