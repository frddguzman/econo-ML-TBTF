"""Single-regime re-renders for eta=0 figures (audit A.3 verbosity strip).

At eta=0 the bilateral-cap denominator (1 - b_j * eta) = 1 collapses,
no bailout payments fire, and the three fiscal regimes produce
identical single-seed trajectories. The audit cross-cut A.3 strip
requires us to show ONE panel per diagnostic at eta=0, not three.

This script re-renders the five eta=0 figures as single-regime versions,
preserving the canonical paths so the LaTeX cross-references resolve
unchanged. The originals are preserved as <name>.bak.png on first run.

Targets:
  1. 3_1_1_brini_canonical_e0.png  (hub formation diagnostic)
  2. 3_1_2_topology_snapshots_e0.png  (topology snapshots)
  3. 3_1_3_ddf_indegree_e0.png  (DDF in-degree)
  4. 3_3_2_ccf_chain_e0.png  (CCFs pairs a-c)
  5. 3_3_2_ccf_fiscal_e0.png  (CCFs pairs d-e)

All five take the no-tax regime as the single-regime reference because
at eta=0 the three regimes are identical and the no-tax data file is
the cleanest source.
"""
import os, sys, shutil, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thesis_render_utils import asset_path, save_figure, setup_mpl, COLORS

setup_mpl()
PROJ = os.path.dirname(os.path.abspath(__file__))


def backup_if_needed(path):
    if not os.path.exists(path):
        return
    bak = path + '.bak.png' if path.endswith('.png') else path + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)


def load_csv(path):
    if not os.path.exists(path):
        return None
    return list(csv.DictReader(open(path, encoding='utf-8')))


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------- 1. Brini hub formation diagnostic (eta=0 single-regime) ----------

def render_brini_e0_single():
    out = asset_path('ch3_1_topology', '3_1_1_brini_canonical_e0.png')
    backup_if_needed(out)
    rows = load_csv('../Simulations/dash_w58_nt_e0.csv')
    if rows is None:
        # Try dashboard fallback via DATA
        rows = _dashboard_rows('w58_nt_e0')
    N = 50

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             sharey=True,
                             gridspec_kw={'wspace': 0.05})
    if rows is None:
        for ax in axes:
            ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=10)
        fig.suptitle(r'Hub formation diagnostic at $\eta = 0$, '
                     'single-regime panel.', fontsize=11, y=0.95)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(fig, out)
        return

    t = np.array([float(r['time']) for r in rows])
    bl = np.array([float(r['best_lender']) for r in rows])
    bl = np.where(bl < 0, np.nan, bl)
    fit = np.array([float(r['best_lender_fitness']) for r in rows])
    fit = np.where(fit < 0, np.nan, fit)
    cli = np.array([float(r['best_lender_clients']) for r in rows])
    cli = np.where(cli < 0, np.nan, cli)

    for col, (t_lo, t_hi) in enumerate([(0, 500), (500, 1000)]):
        ax = axes[col]
        mask = (t >= t_lo) & (t <= t_hi)
        t_w = t[mask]
        ax.plot(t_w, bl[mask] / N, color='black', linewidth=0.7,
                drawstyle='steps-post',
                label='Hub ID' if col == 0 else None)
        ax.plot(t_w, fit[mask], color=COLORS['cGreen'], linewidth=0.6,
                linestyle=':', alpha=0.85,
                label='Fitness' if col == 0 else None)
        ax.plot(t_w, cli[mask] / N, color=COLORS['cRed'], linewidth=0.7,
                linestyle='--', alpha=0.85,
                label='Clients (in-degree)' if col == 0 else None)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Time', fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Single regime\n(normalized)', fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.92), fontsize=10, frameon=False)
    fig.suptitle(r'Hub formation diagnostic at $\eta = 0$, '
                 'single-regime panel (three fiscal regimes coincide).',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    save_figure(fig, out)


def _dashboard_rows(cell):
    """Fallback: pull series from dashboard.html const DATA into a CSV-like
    list-of-dicts."""
    path = os.path.join(PROJ, 'dashboard.html')
    with open(path, encoding='utf-8') as f:
        text = f.read()
    marker = 'const DATA = '
    start = text.find(marker)
    obj_start = start + len(marker)
    depth = 0
    i = obj_start
    in_string = False
    escape = False
    obj_end = 0
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
        elif c == '\\':
            escape = True
        elif c == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    obj_end = i + 1
                    break
        i += 1
    data = json.loads(text[obj_start:obj_end])
    if cell not in data:
        return None
    d = data[cell]
    T = len(d['time'])
    rows = []
    for k in range(T):
        rows.append({
            'time': d['time'][k],
            'best_lender': d['best_lender'][k] if d['best_lender'][k] is not None else -1,
            'best_lender_fitness': d['best_lender_fitness'][k] if d['best_lender_fitness'][k] is not None else -1,
            'best_lender_clients': d['best_lender_clients'][k] if d['best_lender_clients'][k] is not None else -1,
            'bankruptcies_shock': d['bankruptcies_shock'][k],
            'bankruptcies_contagion': d['bankruptcies_contagion'][k],
            'fire_sale_survivors': d['fire_sale_survivors'][k],
            'bankruptcies_fiscal': d['bankruptcies_fiscal'][k],
            'bailout_count': d['bailout_count'][k] if 'bailout_count' in d else 0,
        })
    return rows


# ---------- 2. Topology snapshots eta=0 single-regime ----------

def spring_layout(n_nodes, edges, iterations=80, seed=42):
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=1.0, size=(n_nodes, 2))
    if not edges:
        return pos
    k = 1.0 / np.sqrt(n_nodes)
    for it in range(iterations):
        t = 0.1 * (1 - it / iterations)
        disp = np.zeros_like(pos)
        for i in range(n_nodes):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1) + 1e-6
            force = (k * k) / dist
            disp[i] += np.sum(delta / dist[:, None] * force[:, None], axis=0)
        for (a, b) in edges:
            delta = pos[a] - pos[b]
            dist = np.linalg.norm(delta) + 1e-6
            force = (dist * dist) / k
            disp[a] -= delta / dist * force
            disp[b] += delta / dist * force
        norm = np.linalg.norm(disp, axis=1) + 1e-6
        pos += disp / norm[:, None] * np.minimum(norm, t)[:, None]
    return pos


def render_network(ax, snapshot, title, n_banks=50):
    if snapshot is None:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
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
    in_degree = np.zeros(n_banks)
    for (b, l) in edge_pairs:
        in_degree[l] += 1
    max_deg = max(1, in_degree.max())
    sizes = 8 + 80 * (in_degree / max_deg)
    colors = np.where(in_degree > 0, COLORS['cRed'], COLORS['cGray'])
    if in_degree.max() > 0:
        hub = int(np.argmax(in_degree))
        colors = list(colors)
        colors[hub] = COLORS['cBlue']
    for (b, l) in edge_pairs:
        x0, y0 = pos[b]; x1, y1 = pos[l]
        ax.plot([x0, x1], [y0, y1], color=COLORS['cGray'], linewidth=0.4,
                alpha=0.5)
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=sizes, edgecolors='black',
               linewidths=0.3, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    span = max(pos[:, 0].max() - pos[:, 0].min(),
               pos[:, 1].max() - pos[:, 1].min(), 0.5) * 0.6 + 0.2
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)


def render_topology_e0_single():
    out = asset_path('ch3_1_topology', '3_1_2_topology_snapshots_e0.png')
    backup_if_needed(out)
    d = load_json('../Simulations/topology_w58_nt_e0.json')
    snapshots = {0: None, 500: None, 999: None}
    if d:
        for k in ['0', '500', '999']:
            if k in d:
                snapshots[int(k)] = d[k]

    cols = [(0, 't = 0'),
            (500, 't = 500'),
            (999, 't = 999')]
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 3.6))
    for j, (t, label) in enumerate(cols):
        snap = snapshots.get(t)
        render_network(axes[j], snap, label)
    fig.suptitle(r'Topology snapshots at $\eta = 0$, single-regime panel '
                 '(three fiscal regimes coincide); spring layout, hub in '
                 'blue, lenders in red, periphery grey.',
                 fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out)


# ---------- 3. DDF eta=0 single-regime ----------

def render_ddf_e0_single():
    out = asset_path('ch3_1_topology', '3_1_3_ddf_indegree_e0.png')
    backup_if_needed(out)
    d = load_json('../Simulations/topology_w58_nt_e0.json')
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if not d:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=10)
        fig.suptitle(r'Decumulative in-degree distribution at $\eta = 0$, '
                     'single-regime panel.', fontsize=10, y=0.97)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        save_figure(fig, out)
        return

    # Pool degree counts across snapshots
    degrees = []
    n_banks = 50
    for snap_key in ['0', '500', '999']:
        snap = d.get(snap_key)
        if snap is None:
            continue
        in_deg = np.zeros(n_banks)
        for e in snap.get('edges', []):
            try:
                l = int(e['lender'])
                if 0 <= l < n_banks:
                    in_deg[l] += 1
            except (ValueError, KeyError, TypeError):
                continue
        degrees.extend([int(x) for x in in_deg if x > 0])

    if not degrees:
        ax.text(0.5, 0.5, 'no edges', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=10)
    else:
        deg_sorted = sorted(degrees)
        ks = np.array(sorted(set(deg_sorted)))
        N = len(deg_sorted)
        ccdf = np.array([sum(1 for x in deg_sorted if x >= k) / N
                         for k in ks])
        ax.loglog(ks, ccdf, marker='o', color=COLORS['cBlue'],
                  linewidth=1.2, markersize=6, label='Single regime')
        ax.set_xlabel(r'In-degree $k$', fontsize=10)
        ax.set_ylabel(r'$P(K \geq k)$', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(1e-3, 2.0)

    fig.suptitle(r'Decumulative in-degree distribution at $\eta = 0$, '
                 'single-regime panel (three fiscal regimes coincide).',
                 fontsize=10, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, out)


# ---------- 4. CCF chain eta=0 single-regime (pairs a-c) ----------

def to_arr(seq):
    return np.array([float(v) if v is not None else 0.0 for v in seq])


def ccf(x, y, max_lag):
    sx = x.std(); sy = y.std()
    if sx == 0 or sy == 0:
        return np.arange(-max_lag, max_lag + 1), np.zeros(2 * max_lag + 1)
    x = (x - x.mean()) / sx
    y = (y - y.mean()) / sy
    lags = np.arange(-max_lag, max_lag + 1)
    n = len(x)
    out = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag < 0:
            out[i] = np.mean(x[-lag:] * y[:n + lag])
        elif lag > 0:
            out[i] = np.mean(x[:n - lag] * y[lag:])
        else:
            out[i] = np.mean(x * y)
    return lags, out


def color_for(peak):
    NOISE = 0.062
    a = abs(peak)
    if a > 2 * NOISE:
        return 'darkred'
    if a > NOISE:
        return 'orange'
    return 'gray'


def _draw_ccf_panel(ax, lags, c, ylabel=None):
    NOISE = 0.062
    peak_idx = int(np.argmax(np.abs(c)))
    peak_val = c[peak_idx]; peak_lag = lags[peak_idx]
    col = color_for(peak_val)
    ax.fill_between(lags, c, 0, alpha=0.25, color=col)
    ax.plot(lags, c, color=col, linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axhline(NOISE, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.axhline(-NOISE, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.set_ylim(-0.4, 0.4)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.95,
            f'{peak_val:+.2f} at lag {int(peak_lag):+d}',
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)


def render_ccf_chain_e0_single():
    out = asset_path('ch3_3_claim2_contagion', '3_3_2_ccf_chain_e0.png')
    backup_if_needed(out)
    rows = _dashboard_rows('w58_nt_e0')
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    if rows is None:
        for ax in axes:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=10)
    else:
        shock = to_arr([r['bankruptcies_shock'] for r in rows])
        contag = to_arr([r['bankruptcies_contagion'] for r in rows])
        survivors = to_arr([r['fire_sale_survivors'] for r in rows])
        lags, c1 = ccf(shock, contag, 20)
        _draw_ccf_panel(axes[0], lags, c1, ylabel='shock and contagion')
        _, c2 = ccf(shock, survivors, 20)
        _draw_ccf_panel(axes[1], lags, c2, ylabel='shock and survivors')
        _, c3 = ccf(survivors, contag, 20)
        _draw_ccf_panel(axes[2], lags, c3, ylabel='survivors and contagion')
        axes[2].set_xlabel('lag')
    fig.suptitle(r'Pairs (a)--(c) at $\eta = 0$, single-regime panel '
                 '(three fiscal regimes coincide).' '\n'
                 r'95% noise band $\pm 0.062$; '
                 'gray = noise, orange = marginal, red = strong signal.',
                 fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out)


def render_ccf_fiscal_e0_single():
    out = asset_path('ch3_3_claim2_contagion', '3_3_2_ccf_fiscal_e0.png')
    backup_if_needed(out)
    rows = _dashboard_rows('w58_nt_e0')
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    if rows is None:
        for ax in axes:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=10)
    else:
        contag = to_arr([r['bankruptcies_contagion'] for r in rows])
        fisc = to_arr([r['bankruptcies_fiscal'] for r in rows])
        bail = to_arr([r['bailout_count'] for r in rows])
        lags, c1 = ccf(contag, fisc, 20)
        _draw_ccf_panel(axes[0], lags, c1, ylabel='contagion and fiscal')
        _, c2 = ccf(bail, fisc, 20)
        _draw_ccf_panel(axes[1], lags, c2, ylabel='bailout and fiscal')
        axes[1].set_xlabel('lag')
    fig.suptitle(r'Pairs (d)--(e) at $\eta = 0$, single-regime panel.' '\n'
                 'All panels coincide at constant zero: no bailout event '
                 'fires at this anchor.', fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, out)


def main():
    print('-- brini e0 single')
    render_brini_e0_single()
    print('-- topology e0 single')
    render_topology_e0_single()
    print('-- DDF e0 single')
    render_ddf_e0_single()
    print('-- CCF chain e0 single')
    render_ccf_chain_e0_single()
    print('-- CCF fiscal e0 single')
    render_ccf_fiscal_e0_single()


if __name__ == '__main__':
    main()
