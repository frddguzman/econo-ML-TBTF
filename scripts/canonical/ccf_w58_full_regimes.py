"""Full three-regime CCF analysis at w58 (and bsl reference) using dashboard.html embedded DATA.

Cells (single-seed SEED=26474, dashboard dump):
  - w58_{nt,st,rf}_e01     — default τ=1e-4 (w58 standard config)
  - w58_t5_{nt,st,rf}_e01  — τ=1e-5 (the canonical commit candidate for fund regime)
  - bsl_{nt,st,rf}_e01     — bsl loud-channel reference

Regime labels:
  nt = no_tax (no fiscal feedback)
  st = socialized_tax (post-period tax on survivors)
  rf = resolution_fund (ex-ante levy)

Channel pairs:
  shock → contagion       (claim 1 mechanism)
  contagion → fiscal      (fiscal feedback strength)
  shock → zombies         (claim 2 zombie creation)
  zombies → contagion     (zombie hazard channel)
  bailout_bill → fiscal   (bailout drives fiscal pressure)

Output:
  ccf_w58_full_regimes.csv  — all (cell, pair) peak CCFs
  ccf_w58_full_regimes.png  — 5 pairs × 9 cells grid (grouped by cand: w58, w58_t5, bsl)
"""
import os, sys, json, csv
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.abspath(__file__))
DASH_PATH = os.path.join(PROJ, 'dashboard.html')

REGIMES = ['nt', 'st', 'rf']
REGIME_LABELS = {'nt': 'no_tax', 'st': 'social', 'rf': 'res_fund'}

CANDS = ['w58_t5']  # canonical only (τ=1e-5 fund); nt/st invariant to τ
CAND_LABELS = {'w58_t5': 'w58 cv=0 canonical (τ=1e-5)'}

CELLS = [(c, r, f'{c}_{r}_e01') for c in CANDS for r in REGIMES]

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


def main():
    print('Loading DATA...', flush=True)
    data = extract_data()
    print(f'  loaded {len(data)} cells', flush=True)

    # Compute CCFs
    results = {}
    cell_totals = {}
    for cand, regime, cell in CELLS:
        if cell not in data:
            print(f'  WARN: {cell} missing', flush=True)
            continue
        total_bk = int(sum(to_array(data[cell]['bankruptcy'])))
        total_fiscal = int(sum(to_array(data[cell].get('bankruptcies_fiscal', [0]))))
        total_zomb = int(sum(to_array(data[cell].get('fire_sale_survivors', [0]))))
        total_bill = float(sum(to_array(data[cell].get('bailout_bill', [0]))))
        cell_totals[cell] = (total_bk, total_fiscal, total_zomb, total_bill)
        for pi, (x_name, y_name) in enumerate(CHANNEL_PAIRS):
            x = to_array(data[cell][x_name])
            y = to_array(data[cell][y_name])
            lags, c = ccf(x, y, MAX_LAG)
            peak_idx = int(np.argmax(np.abs(c)))
            results[(cell, pi)] = (lags, c, lags[peak_idx], c[peak_idx], c[MAX_LAG])

    T_ref = len(data[CELLS[0][2]]['time'])
    ci = 1.96 / np.sqrt(T_ref)
    print(f'  T={T_ref}, 95% noise band: |CCF| < {ci:.3f}', flush=True)

    # Print cell totals
    print()
    print('=== Cell totals (informs whether channels are sampling-limited) ===')
    print(f'{"cell":>17} | {"total_bk":>9} {"fiscal":>7} {"zombies":>7} {"bailout":>7}')
    print('-'*60)
    for _, _, cell in CELLS:
        if cell in cell_totals:
            tb, tf, tz, tbi = cell_totals[cell]
            print(f'{cell:>17} | {tb:>9} {tf:>7} {tz:>7} {tbi:>7.0f}')

    # Print summary by cell × pair
    print()
    print('=== CCFs per cell × pair (peak |CCF| / lag / signal strength) ===')
    print(f'{"cell":>17} | {"pair":>22} | {"peak |CCF|":>10} {"lag":>4} | {"@0":>6} | sig')
    print('-'*78)
    csv_rows = []
    for _, _, cell in CELLS:
        for pi, label in enumerate(PAIR_LABELS):
            if (cell, pi) not in results: continue
            _, _, peak_lag, peak_val, zero_val = results[(cell, pi)]
            sig = '★' if abs(peak_val) > 2*ci else ('?' if abs(peak_val) > ci else '·')
            print(f'{cell:>17} | {label:>22} | {peak_val:>+9.3f} {peak_lag:>+4d} | {zero_val:>+5.3f} | {sig}')
            csv_rows.append({
                'cell': cell, 'pair': label, 'peak_ccf': round(peak_val, 4),
                'peak_lag': int(peak_lag), 'ccf_at_0': round(zero_val, 4),
                'noise_ci_95': round(ci, 4),
                'signal': 'strong' if abs(peak_val) > 2*ci else ('marginal' if abs(peak_val) > ci else 'noise'),
            })
        print()

    csv_path = os.path.join(PROJ, 'ccf_w58_full_regimes.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['cell', 'pair', 'peak_ccf', 'peak_lag', 'ccf_at_0', 'noise_ci_95', 'signal'])
        w.writeheader()
        for r in csv_rows: w.writerow(r)
    print(f'CSV: {csv_path}', flush=True)

    # PNG: 5 pairs × 9 cells (3 cands × 3 regimes)
    n_pairs = len(CHANNEL_PAIRS)
    n_cells = len(CELLS)
    fig, axes = plt.subplots(n_pairs, n_cells, figsize=(2.6 * n_cells, 2.0 * n_pairs),
                             sharex=True, sharey=True)
    if n_pairs == 1: axes = axes.reshape(1, n_cells)
    if n_cells == 1: axes = axes.reshape(n_pairs, 1)

    for col, (cand, regime, cell) in enumerate(CELLS):
        for row, label in enumerate(PAIR_LABELS):
            ax = axes[row, col]
            if (cell, row) not in results:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
                continue
            lags, c, peak_lag, peak_val, _ = results[(cell, row)]
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
            if row == 0:
                ax.set_title(f'{cand}\n{REGIME_LABELS[regime]}', fontsize=9)
            if row == n_pairs - 1: ax.set_xlabel('lag', fontsize=8)
            if col == 0: ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.25, linewidth=0.4)

    fig.suptitle(f'Channel CCFs at η=0.1, all 3 fiscal regimes — single-seed SEED=26474\n'
                 f'95% noise band ±{ci:.3f}; gray = noise, orange = marginal, red = strong signal',
                 fontsize=11, y=0.995)
    fig.tight_layout()
    png_path = os.path.join(PROJ, 'ccf_w58_full_regimes.png')
    fig.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {png_path}', flush=True)


if __name__ == '__main__':
    main()
