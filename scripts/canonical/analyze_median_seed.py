"""Per-metric median seed analysis at canonical (w58 cv=0, social, eta=0.10).

Computes per-metric median across 6 seeds (5-seed pool + SEED=26474):
  - total_bk, contagion (mortality)
  - max_ten, avg_ten, turnovers (hub stability)
  - avg_cli (in-degree)
  - mean_equity (equity drift)

Identifies which seed lands closest to median for each metric.
If SEED=26474 dominates, vindicates the OG dashboard convention.
"""
import os, csv, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr


def run_canonical(seed):
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=0.10, regime='socialized_tax')
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T])
    raw_ids = [b for b in bl if b is not None and b >= 0]
    runs = []
    if raw_ids:
        prev = raw_ids[0]; rl = 1
        for k in raw_ids[1:]:
            if k == prev: rl += 1
            else: runs.append(rl); rl = 1; prev = k
        runs.append(rl)
    max_ten = max(runs) if runs else 0
    avg_ten = sum(runs)/len(runs) if runs else 0
    turnovers = max(0, len(runs) - 1)
    blc = [s.best_lender_clients[t] for t in range(T) if s.best_lender_clients[t] is not None and s.best_lender_clients[t] >= 0]
    avg_cli = sum(blc)/len(blc) if blc else 0
    bankruptcy = sum(b for b in s.bankruptcy[:T] if b is not None)
    contagion = sum(b for b in s.bankruptcies_contagion[:T] if b is not None)
    equity_arr = getattr(s, 'equity', None)
    if equity_arr is not None:
        eq_vals = [float(e) for e in equity_arr[:T] if e is not None]
        mean_equity = sum(eq_vals)/len(eq_vals) if eq_vals else 0
        std_equity = float(np.std(eq_vals)) if eq_vals else 0
        final_equity = eq_vals[-1] if eq_vals else 0
    else:
        mean_equity = std_equity = final_equity = 0
    return {
        'seed': seed,
        'total_bk': int(bankruptcy),
        'contagion': int(contagion),
        'max_ten': max_ten,
        'avg_ten': round(avg_ten, 2),
        'turnovers': turnovers,
        'avg_cli': round(avg_cli, 2),
        'mean_equity': round(mean_equity, 1),
        'std_equity': round(std_equity, 1),
        'final_equity': round(final_equity, 1),
    }


def main():
    print('Running 6 sims at canonical (w58 cv=0, social, eta=0.10)...', flush=True)
    seeds = [26462, 26463, 26464, 26465, 26466, 26474]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run_canonical, seeds))
    results.sort(key=lambda x: x['seed'])

    metrics = ['total_bk', 'contagion', 'max_ten', 'avg_ten', 'turnovers', 'avg_cli',
               'mean_equity', 'std_equity', 'final_equity']

    print()
    print('='*120)
    print('Canonical metrics per seed (w58 cv=0, social, eta=0.10)')
    print('='*120)
    hdr = f"{'metric':<14}"
    for r in results:
        hdr += f" | {r['seed']:>8}"
    hdr += " | median seed(s)"
    print(hdr)
    print('-'*120)
    for m in metrics:
        vals = [(r['seed'], r[m]) for r in results]
        sorted_vals = sorted(vals, key=lambda x: x[1])
        # 6 values: median is between positions 2 and 3 (0-indexed)
        median_seeds = [sorted_vals[2][0], sorted_vals[3][0]]
        median_val = (sorted_vals[2][1] + sorted_vals[3][1]) / 2
        line = f'{m:<14}'
        for s, v in vals:
            mark = '*' if s in median_seeds else ' '
            line += f' | {v:>7}{mark}'
        line += f' | {median_seeds[0]}, {median_seeds[1]} (med={median_val:.1f})'
        print(line)

    # Tally: which seed appears as median how often?
    print()
    print('=== Median-seed tally (out of 9 metrics) ===')
    tally = {s: 0 for s in seeds}
    for m in metrics:
        vals = [(r['seed'], r[m]) for r in results]
        sorted_vals = sorted(vals, key=lambda x: x[1])
        median_seeds = [sorted_vals[2][0], sorted_vals[3][0]]
        for s in median_seeds:
            tally[s] += 1
    for s in seeds:
        print(f'  SEED={s}: appears as median in {tally[s]} / 9 metrics')


if __name__ == '__main__':
    main()
