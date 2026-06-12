"""Quick one-off: cand b (bilateral, ω=0.52) + cash boost + various β values.
No CSV dump, no dashboard. Just print max_deg / max tenure / top hub-id distribution.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474

def run_one(basis, omega, beta, cash_boost):
    cfg = ddr.make_config(basis=basis, omega=omega, eta=0.1, regime='socialized_tax')
    cfg['beta'] = beta
    cfg['cash_boost_random'] = cash_boost
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=SEED, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    lender_max = ddr.compute_lender_max(s, T)
    bl = list(s.best_lender[:T])
    counts = Counter(b for b in bl if b >= 0)
    top5 = counts.most_common(5)
    max_indeg = max((s.best_lender_clients[t] for t in range(T) if s.best_lender_clients[t] >= 0), default=0)
    total_bk = int(sum(s.bankruptcy[:T]))
    return lender_max, max_indeg, top5, total_bk

print('cand b (bilateral, omega=0.52), st x eta=0.1, cash_boost on. Sweep beta:')
print(f'{"beta":>4} | {"lend_max_ten":>12} | {"max_indeg":>10} | {"total_bk":>9} | top hubs')
for beta in [5, 10, 20]:
    lt, mi, top, bk = run_one('bilateral', 0.52, beta, True)
    top_str = ' '.join(f'{int(b)}:{c}' for b, c in top)
    print(f'{beta:>4} | {lt:>12} | {mi:>10} | {bk:>9} | {top_str}')

print()
print('cand b (bilateral, omega=0.52), st x eta=0.1, cash_boost OFF (uniform). Sweep beta:')
print(f'{"beta":>4} | {"lend_max_ten":>12} | {"max_indeg":>10} | {"total_bk":>9} | top hubs')
for beta in [5, 10, 20]:
    lt, mi, top, bk = run_one('bilateral', 0.52, beta, False)
    top_str = ' '.join(f'{int(b)}:{c}' for b, c in top)
    print(f'{beta:>4} | {lt:>12} | {mi:>10} | {bk:>9} | {top_str}')
