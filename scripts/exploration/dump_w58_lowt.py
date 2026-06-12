"""Phase 0: dump 9 dashboard cells each for w58_t6 (tau=1e-6) and w58_t5 (tau=1e-5).

Both cands share (mu=0.7, omega=0.58, gamma=0.10) — same as w58 — but with
fund_levy_rate overridden to the sub-realistic value where claim 3 still preserves
in the resolution_fund regime (see updated_contextv5.md section 14.2).

3 regimes x 3 etas = 9 cells per cand x 2 cands = 18 sims at SEED=26474.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474
MU = 0.70
OMEGA = 0.58
GAMMA = 0.10
REGIMES = ddr.REGIME_DEFS
ETAS    = ddr.ETA_DEFS

VARIANTS = [
    ('w58_t6', 1e-6),
    ('w58_t5', 1e-5),
]


def dump_one(args):
    tag, eta, regime, seed, tau = args
    cfg = ddr.make_config(basis='equity', omega=OMEGA, eta=eta, regime=regime)
    cfg['mu'] = MU
    cfg['gamma_capital'] = GAMMA
    cfg['fund_levy_rate'] = tau
    model = interbank.Model()
    model.test = True
    model.configure(**cfg)
    model.config.lender_change = lc.determine_algorithm("Boltzmann")
    model.initialize(seed=seed, generate_plots=False)
    model.simulate_full()
    model.finish()
    T = model.t
    stats = model.statistics

    columns = {'time': list(range(T))}
    for name in ddr.STAT_NAMES_EXT:
        columns[name] = ddr.extract_array(stats, name, T)
    header = ['time'] + ddr.STAT_NAMES_EXT
    rows = [','.join(header)]
    for t in range(T):
        rows.append(','.join([str(t)] + [ddr.safe_val(columns[name][t])
                                          for name in ddr.STAT_NAMES_EXT]))
    csv_path = os.path.join(ddr.SAVE_DIR, f'dash_{tag}.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))

    bd_rows = getattr(stats, '_bank_rows', []) or []
    bd_lines = [','.join(ddr.BANK_DETAIL_COLS)]
    for r in bd_rows:
        bd_lines.append(','.join(ddr.safe_val(r.get(c)) for c in ddr.BANK_DETAIL_COLS))
    bd_path = os.path.join(ddr.SAVE_DIR, f'bank_detail_{tag}.csv')
    with open(bd_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bd_lines))

    return tag, T, ddr.compute_lender_max(stats, T), len(bd_rows)


def main():
    jobs = []
    for cand_tag, tau in VARIANTS:
        for r_tag, regime in REGIMES:
            for e_tag, eta in ETAS:
                tag = f'{cand_tag}_{r_tag}_{e_tag}'
                jobs.append((tag, eta, regime, SEED, tau))
    workers = max(1, min(len(jobs), 6))
    print(f'w58_t6/w58_t5 cells: {len(jobs)} sims, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, T, lmax, n_bd in pool.map(dump_one, jobs):
            print(f'  {tag}: T={T} lender_max={lmax} bank_rows={n_bd}')
    print('\nDone. Add cands to CAND_DEFS and regenerate dashboard.')


if __name__ == '__main__':
    main()
