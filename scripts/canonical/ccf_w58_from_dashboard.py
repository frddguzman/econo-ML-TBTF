"""CCF analysis at w58 (and bsl reference) using existing dashboard.html embedded DATA.
No new sims needed — single-seed (SEED=26474) per dashboard dump convention.

Computes cross-correlation functions between channel time-series at:
  - w58_st_e01 (w58 social η=0.1) — the supervisor anchor cell
  - w58_t5_st_e01 (w58 with τ=1e-5, the candidate canonical for fund regime)
  - bsl_st_e01 (bsl social η=0.1, loud-channel reference)

Channel pairs:
  - shock → contagion  (does shock predict contagion?)
  - contagion → fiscal (does contagion drive fiscal pressure?)
  - shock → zombies (zombie creation)
  - zombies → contagion (zombie hazard channel)
  - bailout_bill → fiscal (bailout cost → fiscal deaths)

Output:
  - ccf_w58_dashboard.csv — per-pair peak CCF + lag for each cell
  - ccf_w58_dashboard.png — visual: 3 cells × 5 pairs grid of CCF plots
"""
import os, sys, json, csv
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.abspath(__file__))
DASH_PATH = os.path.join(PROJ, 'dashboard.html')

CELLS = ['w58_st_e01', 'w58_t5_st_e01', 'bsl_st_e01']

CHANNEL_PAIRS = [
    ('bankruptcies_shock', 'bankruptcies_contagion'),
    ('bankruptcies_contagion', 'bankruptcies_fiscal'),
    ('bankruptcies_shock', 'fire_sale_survivors'),
    ('fire_sale_survivors', 'bankruptcies_contagion'),
    ('bailout_bill', 'bankruptcies_fiscal'),
]
PAIR_LABELS = [
    'shock → contagion',
    'contagion → fiscal',
    'shock → zombies',
    'zombies → contagion',
    'bailout → fiscal',
]
MAX_LAG = 20


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
    return json.loads(text[obj_start:obj_end])


def to_array(seq):
    """Convert list with possible None to numpy array."""
    return np.array([float(v) if v is not None else 0.0 for v in seq])


def ccf(x, y, max_lag):
    """Cross-correlation function (Pearson normalized).
    Positive lag = x leads y by lag steps.
    """
    x = (x - x.mean()) / (x.std() if x.std() > 0 else 1)
    y = (y - y.mean()) / (y.std() if y.std() > 0 else 1)
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


def main():
    print('Loading DATA from dashboard.html...', flush=True)
    data = extract_data()
    print(f'  loaded {len(data)} cells', flush=True)

    # Compute CCFs for each (cell, pair)
    results = {}  # (cell, pair_idx) -> (lags, ccf, peak_lag, peak_val, ccf_at_0)
    for cell in CELLS:
        if cell not in data:
            print(f'  WARN: {cell} not in DATA', flush=True)
            continue
        for pi, (x_name, y_name) in enumerate(CHANNEL_PAIRS):
            x = to_array(data[cell][x_name])
            y = to_array(data[cell][y_name])
            lags, c = ccf(x, y, MAX_LAG)
            peak_idx = int(np.argmax(np.abs(c)))
            results[(cell, pi)] = (lags, c, lags[peak_idx], c[peak_idx], c[MAX_LAG])

    # 95% noise band
    T_ref = len(data[CELLS[0]]['time'])
    ci = 1.96 / np.sqrt(T_ref)
    print(f'  T={T_ref}, 95% noise band: |CCF| < {ci:.3f}', flush=True)
    print()

    # Print summary
    print('=== Channel CCFs (single-seed SEED=26474, dashboard dump) ===')
    print(f'{"cell":>14} | {"pair":>22} | {"peak |CCF|":>10} {"peak lag":>9} | {"|CCF| at 0":>10} | {"signal":>7}')
    print('-'*90)
    csv_rows = []
    for cell in CELLS:
        for pi, label in enumerate(PAIR_LABELS):
            if (cell, pi) not in results: continue
            _, _, peak_lag, peak_val, zero_val = results[(cell, pi)]
            signal = '★' if abs(peak_val) > 2*ci else ('?' if abs(peak_val) > ci else '·')
            print(f'{cell:>14} | {label:>22} | {peak_val:>+9.3f}  {peak_lag:>+8d}  | {zero_val:>+9.3f} | {signal:>7}')
            csv_rows.append({
                'cell': cell, 'pair': label, 'peak_ccf': round(peak_val, 4),
                'peak_lag': int(peak_lag), 'ccf_at_0': round(zero_val, 4),
                'noise_ci_95': round(ci, 4),
                'signal_strength': 'strong' if abs(peak_val) > 2*ci else ('marginal' if abs(peak_val) > ci else 'noise'),
            })

    # CSV
    csv_path = os.path.join(PROJ, 'ccf_w58_dashboard.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['cell', 'pair', 'peak_ccf', 'peak_lag', 'ccf_at_0', 'noise_ci_95', 'signal_strength'])
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f'\nCSV: {csv_path}', flush=True)

    # Plot grid: pairs as rows, cells as cols
    n_pairs = len(CHANNEL_PAIRS)
    n_cells = len(CELLS)
    fig, axes = plt.subplots(n_pairs, n_cells, figsize=(4 * n_cells, 2.2 * n_pairs), sharex=True, sharey=True)
    if n_pairs == 1: axes = axes.reshape(1, n_cells)
    if n_cells == 1: axes = axes.reshape(n_pairs, 1)
    for row, label in enumerate(PAIR_LABELS):
        for col, cell in enumerate(CELLS):
            ax = axes[row, col]
            if (cell, row) not in results:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue
            lags, c, _, peak_val, _ = results[(cell, row)]
            color = 'darkred' if abs(peak_val) > 2*ci else ('orange' if abs(peak_val) > ci else 'gray')
            ax.fill_between(lags, c, 0, alpha=0.2, color=color)
            ax.plot(lags, c, color=color, linewidth=1.5)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
            ax.axhline(ci, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
            ax.axhline(-ci, color='red', linewidth=0.5, linestyle='--', alpha=0.6)
            ax.set_ylim(-0.4, 0.4)
            if row == 0: ax.set_title(cell, fontsize=10)
            if row == n_pairs - 1: ax.set_xlabel('lag')
            if col == 0: ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3)
            ax.text(0.02, 0.95, f'peak={peak_val:+.3f} @ {results[(cell, row)][2]:+d}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top')
    fig.suptitle(f'Channel CCFs at η=0.1 (social) — single-seed SEED=26474 dashboard data\n95% noise band ±{ci:.3f}; gray = noise · orange = marginal · red = strong signal',
                 fontsize=11)
    fig.tight_layout()
    png_path = os.path.join(PROJ, 'ccf_w58_dashboard.png')
    fig.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {png_path}', flush=True)


if __name__ == '__main__':
    main()
