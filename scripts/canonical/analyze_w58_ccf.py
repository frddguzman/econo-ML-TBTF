"""CCF analysis at w58 cv=0 — does the channel decomposition carry signal at total_bk~2,800?

Runs 5 seeds at w58 cv=0 (mu=0.7, omega=0.58, gamma=0.10) social regime, eta ∈ {0, 0.1},
single-threaded so it doesn't compete with D0-A's 6-worker sweep.

Computes pairwise cross-correlation functions between channel time-series:
  - shock ↔ contagion (does shock predict contagion?)
  - contagion ↔ fiscal (does contagion drive fiscal pressure?)
  - bailout_bill ↔ fiscal (does bailout cost cause fiscal deaths?)
  - shock ↔ zombies (does shock create zombies?)
  - zombies ↔ contagion (zombie channel hypothesis)

For comparison, runs the same at bsl cv=0 (loud-channel reference).

CCF strength criterion: peak |CCF| > 0.15 with magnitude similar to bsl ⇒ real signal.

Output:
  - ccf_w58_st_e01.csv  — per-channel-pair peak CCF + lag, mean across seeds
  - ccf_w58_st_e01.png  — visual CCF plots, w58 vs bsl side-by-side
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEEDS = [26462, 26463, 26464, 26465, 26466]
ETA = 0.10
MAX_LAG = 20
PROJ = os.path.dirname(os.path.abspath(__file__))

CHANNEL_PAIRS = [
    ('shock', 'contagion'),
    ('contagion', 'fiscal_deaths'),
    ('bailout_bill', 'fiscal_deaths'),
    ('shock', 'zombies'),
    ('zombies', 'contagion'),
]

CELLS = [
    ('w58', 0.70, 0.58, 0.10),
    ('bsl', 0.70, 0.50, 0.10),
]


def run_one(cand, mu, omega, gamma, eta, seed):
    cfg = ddr.make_config(basis='equity', omega=omega, eta=eta, regime='socialized_tax')
    cfg['mu'] = mu
    cfg['gamma_capital'] = gamma
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    # Per-period channel time-series
    series = {}
    for name, attr in [('shock', 'bankruptcies_shock'),
                       ('rationing', 'bankruptcies_rationing'),
                       ('repay', 'bankruptcies_repay'),
                       ('contagion', 'bankruptcies_contagion'),
                       ('fiscal_deaths', 'bankruptcies_fiscal'),
                       ('zombies', 'fire_sale_survivors'),
                       ('bailout_bill', 'bailout_bill'),
                       ('total_bk', 'bankruptcy')]:
        arr = getattr(s, attr, None)
        if arr is None:
            series[name] = np.zeros(T)
        else:
            series[name] = np.array([float(v) if v is not None else 0.0 for v in arr[:T]])
    return T, series


def ccf(x, y, max_lag):
    """Cross-correlation function. Returns (lags, ccf_values).
    Positive lag = y at time t, x at time t-lag (i.e., x leads y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
    print(f'CCF analysis at eta={ETA} for w58 cv=0 vs bsl cv=0 (5 seeds each, single-threaded)', flush=True)
    print(f'  channel pairs: {CHANNEL_PAIRS}', flush=True)
    print()

    results = {}  # (cand, seed) -> (T, series_dict)
    for cand, mu, omega, gamma in CELLS:
        for seed in SEEDS:
            print(f'  running {cand} seed={seed}...', flush=True, end=' ')
            T, series = run_one(cand, mu, omega, gamma, ETA, seed)
            results[(cand, seed)] = (T, series)
            print(f'T={T} total_bk={int(series["total_bk"].sum())}', flush=True)

    print()
    print('=== Pairwise CCF peak (mean across 5 seeds) ===')
    print(f'{"cand":>5} {"x":>14} {"y":>14} | {"peak |CCF|":>10} {"peak lag":>8} | {"|CCF| at 0":>10}')
    print('-'*80)
    rows_csv = []
    ccfs_for_plot = {}
    for cand, mu, omega, gamma in CELLS:
        for x_name, y_name in CHANNEL_PAIRS:
            ccf_array_per_seed = []
            for seed in SEEDS:
                T, series = results[(cand, seed)]
                _, c = ccf(series[x_name], series[y_name], MAX_LAG)
                ccf_array_per_seed.append(c)
            ccf_mean = np.mean(ccf_array_per_seed, axis=0)
            ccf_std = np.std(ccf_array_per_seed, axis=0)
            lags = np.arange(-MAX_LAG, MAX_LAG + 1)
            peak_idx = np.argmax(np.abs(ccf_mean))
            peak_lag = lags[peak_idx]
            peak_val = ccf_mean[peak_idx]
            zero_val = ccf_mean[MAX_LAG]
            print(f'{cand:>5} {x_name:>14} {y_name:>14} | {peak_val:>+9.3f}  {peak_lag:>+7d}  | {zero_val:>+9.3f}')
            rows_csv.append({
                'cand': cand, 'x': x_name, 'y': y_name,
                'peak_ccf': round(peak_val, 4), 'peak_lag': int(peak_lag),
                'ccf_at_0': round(zero_val, 4),
                'mean_ccf_std_at_peak': round(ccf_std[peak_idx], 4),
            })
            ccfs_for_plot[(cand, x_name, y_name)] = (lags, ccf_mean, ccf_std)

    # Write CSV
    csv_path = os.path.join(PROJ, 'ccf_w58_st_e01.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['cand', 'x', 'y', 'peak_ccf', 'peak_lag', 'ccf_at_0', 'mean_ccf_std_at_peak'])
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)
    print(f'\nCSV: {csv_path}', flush=True)

    # Plot — one row per channel pair, w58 left, bsl right
    n_pairs = len(CHANNEL_PAIRS)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 2.5 * n_pairs), sharex=True)
    if n_pairs == 1: axes = axes.reshape(1, 2)
    for row, (x_name, y_name) in enumerate(CHANNEL_PAIRS):
        for col, cand in enumerate(['w58', 'bsl']):
            ax = axes[row, col]
            lags, mean, std = ccfs_for_plot[(cand, x_name, y_name)]
            ax.fill_between(lags, mean - std, mean + std, alpha=0.25, color='steelblue')
            ax.plot(lags, mean, color='steelblue', linewidth=1.5)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
            # 95% CI bound for noise: ~1.96/sqrt(T)
            T_ref = list(results.values())[0][0]
            ci = 1.96 / np.sqrt(T_ref)
            ax.axhline(ci, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
            ax.axhline(-ci, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
            ax.set_ylim(-0.4, 0.4)
            ax.set_title(f'{cand}: {x_name} → {y_name}', fontsize=10)
            if row == n_pairs - 1: ax.set_xlabel('lag')
            if col == 0: ax.set_ylabel('CCF')
            ax.grid(alpha=0.3)
    fig.suptitle(f'Channel CCFs at eta={ETA}, social regime, 5-seed mean ± std\n(red dashed = 95% noise band; |peak| above line = signal)', fontsize=11)
    fig.tight_layout()
    png_path = os.path.join(PROJ, 'ccf_w58_st_e01.png')
    fig.savefig(png_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG: {png_path}', flush=True)


if __name__ == '__main__':
    main()
