"""CCF of hub client count vs (shock, contagion) bankruptcy channels.

Phase 6.3 REDO — produces the two §3.5 CCF figures that LD14 of the audit
specifies but that the prior pass left as 1x1 placeholder PNGs:

  thesis_assets/ch3_3_claim2_contagion/3_5_5_ccf_client_count_shock.png
  thesis_assets/ch3_3_claim2_contagion/3_5_5_ccf_client_count_contagion.png

Each figure is a 3 (eta) x 3 (regime) grid of Pearson CCFs at +/- 20-period
lag window over the 1000-period canonical w58 single-seed run (SEED=26474)
dumped into dashboard.html const DATA. Eta cells: 0, 0.10, 0.85 (suffix
_e0, _e01, _e085 in the dashboard cell keys). Regimes: nt = no tax,
st = ex-post tax (social tax), rf = ex-ante levy (reserve fund).

Series:
  - best_lender_clients (hub client count, period-by-period in-degree of
    the best lender)
  - bankruptcies_shock (deposit-shock bankruptcies per period)
  - bankruptcies_contagion (contagion bankruptcies per period)

Output style matches 3_3_2_ccf_chain.png: gray = noise, orange = marginal,
red = strong; 95%% noise band +/- 0.062 at T = 1000.
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.join(PROJ, 'dashboard.html')
OUT_DIR = os.path.join(PROJ, 'thesis_assets', 'ch3_3_claim2_contagion')

ETAS = [('e0', r'$\eta = 0$'), ('e01', r'$\eta = 0.10$'), ('e085', r'$\eta = 0.85$')]
REGIMES = [('nt', 'no tax'), ('st', 'ex-post'), ('rf', 'ex-ante')]

MAX_LAG = 20
NOISE_CI = 0.062  # 95% noise band at T = 1000


def load_data():
    with open(DASH, 'r', encoding='utf-8') as f:
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
    return json.loads(text[obj_start:obj_end])


def to_arr(seq):
    return np.array([float(v) if v is not None else 0.0 for v in seq])


def ccf(x, y, max_lag):
    """Pearson cross-correlation. Positive lag: x leads y by lag steps."""
    sx = x.std()
    sy = y.std()
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
    a = abs(peak)
    if a > 2 * NOISE_CI:
        return 'darkred'
    if a > NOISE_CI:
        return 'orange'
    return 'gray'


def render_grid(data, channel_field, channel_label_text, out_path,
                fig_title_channel):
    """Render a 3 (eta rows) x 3 (regime cols) CCF grid."""
    fig, axes = plt.subplots(len(ETAS), len(REGIMES),
                             figsize=(4 * len(REGIMES), 2.3 * len(ETAS)),
                             sharex=True, sharey=True)
    if len(ETAS) == 1:
        axes = axes.reshape(1, -1)
    if len(REGIMES) == 1:
        axes = axes.reshape(-1, 1)

    summary_rows = []  # for return / print
    for r, (esfx, elbl) in enumerate(ETAS):
        for c, (rsfx, rlbl) in enumerate(REGIMES):
            cell = f'w58_{rsfx}_{esfx}'
            ax = axes[r, c]
            if cell not in data:
                ax.text(0.5, 0.5, f'no cell\n{cell}', ha='center', va='center',
                        transform=ax.transAxes)
                summary_rows.append((esfx, rsfx, None, None))
                continue
            x = to_arr(data[cell]['best_lender_clients'])
            y = to_arr(data[cell][channel_field])
            lags, ccf_vals = ccf(x, y, MAX_LAG)
            peak_idx = int(np.argmax(np.abs(ccf_vals)))
            peak_val = ccf_vals[peak_idx]
            peak_lag = lags[peak_idx]
            col = color_for(peak_val)
            ax.fill_between(lags, ccf_vals, 0, alpha=0.25, color=col)
            ax.plot(lags, ccf_vals, color=col, linewidth=1.5)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
            ax.axhline(NOISE_CI, color='red', linewidth=0.5, linestyle='--',
                       alpha=0.6)
            ax.axhline(-NOISE_CI, color='red', linewidth=0.5, linestyle='--',
                       alpha=0.6)
            ax.set_ylim(-0.4, 0.4)
            ax.grid(alpha=0.3)
            ax.text(0.02, 0.95,
                    f'{peak_val:+.2f} at lag {int(peak_lag):+d}',
                    transform=ax.transAxes, fontsize=8.5,
                    verticalalignment='top', fontweight='bold')
            if r == 0:
                ax.set_title(rlbl, fontsize=11)
            if r == len(ETAS) - 1:
                ax.set_xlabel('lag')
            if c == 0:
                ax.set_ylabel(elbl, fontsize=10)
            summary_rows.append((esfx, rsfx, peak_val, peak_lag))

    title = (f'CCF: hub client count and {fig_title_channel} '
             f'(single-seed, $T = 1000$)\n'
             f'95% noise band $\\pm 0.062$; '
             f'gray = noise, orange = marginal, red = strong signal')
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return summary_rows


def main():
    print('Loading dashboard DATA...', flush=True)
    data = load_data()
    print(f'  loaded {len(data)} cells', flush=True)

    pairs = [
        ('bankruptcies_shock', 'shock channel',
         os.path.join(OUT_DIR, '3_5_5_ccf_client_count_shock.png')),
        ('bankruptcies_contagion', 'contagion channel',
         os.path.join(OUT_DIR, '3_5_5_ccf_client_count_contagion.png')),
        ('bankruptcies_rationing', 'rationing channel',
         os.path.join(OUT_DIR, '3_5_5_ccf_client_count_rationing.png')),
        ('fire_sale_survivors', 'fire-sale-survivor channel',
         os.path.join(OUT_DIR, '3_5_5_ccf_client_count_firesale_survivors.png')),
    ]

    all_summary = {}
    for channel_field, label, out_path in pairs:
        print(f'\n=== {channel_field} ===', flush=True)
        rows = render_grid(data, channel_field, label, out_path, label)
        all_summary[channel_field] = rows
        print(f'  -> {out_path}', flush=True)
        for esfx, rsfx, peak, lag in rows:
            if peak is None:
                print(f'    eta={esfx} regime={rsfx}: missing cell')
            else:
                signal = ('strong' if abs(peak) > 2 * NOISE_CI
                          else 'marginal' if abs(peak) > NOISE_CI
                          else 'noise')
                print(f'    eta={esfx} regime={rsfx}: peak={peak:+.3f} '
                      f'at lag {int(lag):+d} [{signal}]')

    # Plain text summary file too (for prose-writing reference)
    summ_path = os.path.join(PROJ,
                             '_ccf_hub_client_count_summary.txt')
    with open(summ_path, 'w', encoding='utf-8') as f:
        for ch, rows in all_summary.items():
            f.write(f'=== {ch} ===\n')
            for esfx, rsfx, peak, lag in rows:
                if peak is None:
                    f.write(f'  eta={esfx} regime={rsfx}: missing\n')
                    continue
                signal = ('strong' if abs(peak) > 2 * NOISE_CI
                          else 'marginal' if abs(peak) > NOISE_CI
                          else 'noise')
                f.write(f'  eta={esfx} regime={rsfx}: '
                        f'peak={peak:+.4f} at lag {int(lag):+d} [{signal}]\n')
            f.write('\n')
    print(f'\nsummary -> {summ_path}', flush=True)


if __name__ == '__main__':
    main()
