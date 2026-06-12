"""§3.1 — CCF split figures at η=0.85 (above selectivity flip η†=5/6).

Mirrors render_ccf_split.py at η=0.85. Reads dashboard.html const DATA cells
`w58_t5_{nt,st,rf}_e085`. Outputs CCF chain + fiscal figures with η=0.85 caption.

Output: thesis_assets/ch3_3_claim2_contagion/3_3_2_ccf_{chain,fiscal}_e085.png
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.abspath(__file__))
DASH_PATH = os.path.join(PROJ, 'dashboard.html')
OUT_DIR = os.path.join(PROJ, 'thesis_assets', 'ch3_3_claim2_contagion')
os.makedirs(OUT_DIR, exist_ok=True)

ETA_TAG = 'e085'
ETA_LABEL = '0.85'

REGIMES = ['nt', 'st', 'rf']
REGIME_LABELS = {'nt': 'no tax', 'st': 'ex-post', 'rf': 'ex-ante'}
CANDS = ['w58_t5']
CELLS = [(c, r, f'{c}_{r}_{ETA_TAG}') for c in CANDS for r in REGIMES]

CHANNEL_PAIRS = [
    ('bankruptcies_shock', 'bankruptcies_contagion'),
    ('bankruptcies_contagion', 'bankruptcies_fiscal'),
    ('bankruptcies_shock', 'fire_sale_survivors'),
    ('fire_sale_survivors', 'bankruptcies_contagion'),
    ('bailout_bill', 'bankruptcies_fiscal'),
]
PAIR_LABELS = [
    'shock $\\to$ contagion',
    'contagion $\\to$ fiscal',
    'shock $\\to$ survivors',
    'survivors $\\to$ contagion',
    'bailout $\\to$ fiscal',
]
MAX_LAG = 20
CHAIN_PAIRS = [0, 2, 3]
FISCAL_PAIRS = [1, 4]


def extract_data():
    with open(DASH_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    marker = 'const DATA = '
    start = text.find(marker)
    obj_start = start + len(marker)
    depth = 0
    i = obj_start
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
        elif c == "\\":
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
    return json.loads(text[obj_start:obj_end])


def to_array(seq):
    return np.array([float(v) if v is not None else 0.0 for v in seq])


def ccf(x, y, max_lag):
    sx = x.std(); sy = y.std()
    if sx == 0 or sy == 0:
        return np.arange(-max_lag, max_lag+1), np.zeros(2*max_lag+1)
    x = (x - x.mean()) / sx
    y = (y - y.mean()) / sy
    lags = np.arange(-max_lag, max_lag + 1)
    n = len(x)
    out = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        if lag < 0:
            out[i] = np.mean(x[-lag:] * y[:n+lag])
        elif lag > 0:
            out[i] = np.mean(x[:n-lag] * y[lag:])
        else:
            out[i] = np.mean(x * y)
    return lags, out


def render_subset(pair_indices, results, ci, out_path, suptitle):
    n_pairs = len(pair_indices)
    n_cells = len(CELLS)
    fig, axes = plt.subplots(n_pairs, n_cells, figsize=(2.6 * n_cells, 2.0 * n_pairs),
                             sharex=True, sharey=True)
    if n_pairs == 1: axes = axes.reshape(1, n_cells)
    if n_cells == 1: axes = axes.reshape(n_pairs, 1)

    for col, (cand, regime, cell) in enumerate(CELLS):
        for row_idx, pair_idx in enumerate(pair_indices):
            ax = axes[row_idx, col]
            if (cell, pair_idx) not in results:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue
            lags, c, peak_lag, peak_val, _ = results[(cell, pair_idx)]
            color = 'darkred' if abs(peak_val) > 2*ci else ('orange' if abs(peak_val) > ci else 'gray')
            ax.fill_between(lags, c, 0, alpha=0.25, color=color)
            ax.plot(lags, c, color=color, linewidth=1.5)
            ax.axhline(0, color='gray', linewidth=0.4)
            ax.axvline(0, color='gray', linewidth=0.4, linestyle=':')
            ax.axhline(ci, color='red', linewidth=0.4, linestyle='--', alpha=0.5)
            ax.axhline(-ci, color='red', linewidth=0.4, linestyle='--', alpha=0.5)
            ax.set_ylim(-0.4, 0.4)
            ax.tick_params(labelsize=7)
            ax.text(0.03, 0.96, f'{peak_val:+.2f}@{peak_lag:+d}',
                    transform=ax.transAxes, fontsize=7, va='top', fontweight='bold')
            if row_idx == 0:
                ax.set_title(REGIME_LABELS[regime], fontsize=10)
            if row_idx == n_pairs - 1:
                ax.set_xlabel('lag', fontsize=8)
            if col == 0:
                ax.set_ylabel(PAIR_LABELS[pair_idx], fontsize=8)
            ax.grid(alpha=0.25, linewidth=0.4)

    fig.suptitle(suptitle, fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}', flush=True)


def main():
    print('Loading DATA from dashboard.html...', flush=True)
    data = extract_data()
    print(f'  loaded {len(data)} cells', flush=True)

    results = {}
    for cand, regime, cell in CELLS:
        if cell not in data:
            print(f'  WARN: {cell} missing', flush=True)
            continue
        for pi, (x_name, y_name) in enumerate(CHANNEL_PAIRS):
            x = to_array(data[cell][x_name])
            y = to_array(data[cell][y_name])
            lags, c = ccf(x, y, MAX_LAG)
            peak_idx = int(np.argmax(np.abs(c)))
            results[(cell, pi)] = (lags, c, lags[peak_idx], c[peak_idx], c[MAX_LAG])

    T_ref = len(data[CELLS[0][2]]['time'])
    ci = 1.96 / np.sqrt(T_ref)
    print(f'  T={T_ref}, 95% noise band: |CCF| < {ci:.3f}', flush=True)

    chain_path = os.path.join(OUT_DIR, f'3_3_2_ccf_chain_{ETA_TAG}.png')
    chain_title = (f'Channel CCFs: shock-to-contagion chain at $\\eta={ETA_LABEL}$ '
                   f'(single-seed SEED${{}}=26474$)\n'
                   f'95% noise band $\\pm{ci:.3f}$; '
                   f'gray = noise, orange = marginal, red = strong signal')
    render_subset(CHAIN_PAIRS, results, ci, chain_path, chain_title)

    fiscal_path = os.path.join(OUT_DIR, f'3_3_2_ccf_fiscal_{ETA_TAG}.png')
    fiscal_title = (f'Channel CCFs: fiscal-row architecture at $\\eta={ETA_LABEL}$ '
                    f'(single-seed SEED${{}}=26474$)\n'
                    f'95% noise band $\\pm{ci:.3f}$')
    render_subset(FISCAL_PAIRS, results, ci, fiscal_path, fiscal_title)

    print('done.')


if __name__ == '__main__':
    main()
