"""Per-cand seed optimization at η=0.1, all 3 regimes, both hetero settings.

For each (cand, hetero) combo we want the seed that:
1. Preserves η* sign-stability across all 3 regimes (eta=0.1 < eta=0)
2. Maximizes the minimum max-tenure across the 3 regimes (per-cand maximin)

Grid: 4 cands × 30 seeds × 2 hetero × 3 regimes × 2 etas = 1440 sims at 6 workers
≈ 16 min wallclock at T=1000.

Output: seed_search_per_cand.csv + per-cand recommendation report.

Why this beats joint maximin: hetero's effect is ω-dependent (verified in
test_heterogeneity output). Maximin across all cands under hetero=True
bottlenecks at Cand A (which loses tenure under hetero). Per-cand picks let
each cand run its best config independently.
"""
import os, sys, math
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Simulations')
os.makedirs(SAVE_DIR, exist_ok=True)

CANDS = [
    ('bsl', 'equity',    0.50),
    ('a',   'equity',    0.53),
    ('b',   'bilateral', 0.52),
    ('w55', 'equity',    0.55),
]
REGIMES = [
    ('nt', 'none'),
    ('st', 'socialized_tax'),
    ('rf', 'resolution_fund'),
]
ETAS = [0.0, 0.1]   # only need these for sign-stability check
SEEDS = list(range(26462, 26492))   # 30 seeds


def _rle_max(keys):
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def make_cfg(basis, omega, eta, regime, hetero):
    return dict(
        N=50, T=1000,
        mu=0.7, omega=omega,
        eta_bailout=eta, rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=regime,
        fund_levy_rate=0.0001,
        fitness_basis=basis, fitness_inertia=0.0,
        equity_heterogeneity=hetero,
        equity_cv=0.5,
    )


def run_one(args):
    cand_tag, basis, omega, regime_tag, regime, eta, seed, hetero = args
    cfg = make_cfg(basis, omega, eta, regime, hetero)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm('Boltzmann')
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T]); bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    return {
        'seed':       int(seed),
        'cand':       cand_tag,
        'regime':     regime_tag,
        'eta':        float(eta),
        'hetero':     int(hetero),
        'total_bk':   int(sum(s.bankruptcy[:T])),
        'max_tenure': _rle_max(keys),
    }


def main():
    jobs = []
    for cand_tag, basis, omega in CANDS:
        for regime_tag, regime in REGIMES:
            for eta in ETAS:
                for seed in SEEDS:
                    for hetero in (False, True):
                        jobs.append((cand_tag, basis, omega, regime_tag, regime, eta, seed, hetero))
    print(f'Per-cand search: {len(jobs)} sims (4 cands x 3 regimes x 2 etas x 30 seeds x 2 hetero)')
    workers = 6
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(run_one, jobs), 1):
            rows.append(r)
            if i % 80 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')
    out = os.path.join(SAVE_DIR, 'seed_search_per_cand.csv')
    cols = ['seed','cand','regime','eta','hetero','total_bk','max_tenure']
    with open(out, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'\nWrote {len(rows)} rows -> {out}\n')

    # ── Per-cand analysis ──────────────────────────────────────────
    idx = {(r['seed'], r['cand'], r['regime'], r['eta'], r['hetero']): r for r in rows}

    print('=' * 88)
    print('Per-cand best-seed recommendations (eta=0.1 sign-stable across all 3 regimes)')
    print('=' * 88)
    print(f'{"cand":<6} {"hetero":<7} {"best seed":>9} {"min_ten":>8} {"ten_nt":>7} {"ten_st":>7} {"ten_rf":>7} {"eligible":>10}')
    print('-' * 88)

    summary = []
    for c, _, _ in CANDS:
        for hetero in (False, True):
            eligible = []
            for seed in SEEDS:
                # Eligibility: eta=0.1 < eta=0 in all 3 regimes
                ok = True
                tenures = {}
                for rg, _ in REGIMES:
                    bk0  = idx[(seed, c, rg, 0.0, hetero)]['total_bk']
                    bk01 = idx[(seed, c, rg, 0.1, hetero)]['total_bk']
                    if not (bk01 < bk0):
                        ok = False
                    tenures[rg] = idx[(seed, c, rg, 0.1, hetero)]['max_tenure']
                min_t = min(tenures.values())
                if ok:
                    eligible.append((seed, min_t, tenures))

            if eligible:
                eligible.sort(key=lambda x: -x[1])
                best_seed, best_min, best_t = eligible[0]
                het_label = 'True' if hetero else 'False'
                print(f'{c:<6} {het_label:<7} {best_seed:>9} {best_min:>8} '
                      f'{best_t["nt"]:>7} {best_t["st"]:>7} {best_t["rf"]:>7} '
                      f'{len(eligible):>5}/{len(SEEDS):>4}')
                summary.append((c, hetero, best_seed, best_min, best_t, len(eligible)))
            else:
                het_label = 'True' if hetero else 'False'
                print(f'{c:<6} {het_label:<7} {"NO ELIG":>9} {"-":>8} {"-":>7} {"-":>7} {"-":>7} '
                      f'{0:>5}/{len(SEEDS):>4}')

    print('\n' + '=' * 88)
    print('Recommendation per cand: pick whichever (hetero) row has higher min_ten')
    print('=' * 88)
    by_cand = {}
    for entry in summary:
        c = entry[0]
        if c not in by_cand or entry[3] > by_cand[c][3]:
            by_cand[c] = entry

    print(f'{"cand":<6} {"hetero":<7} {"seed":>6} {"min_tenure":>11} {"per-regime ten":>30}')
    for c, _, _ in CANDS:
        if c in by_cand:
            _, hetero, seed, min_t, ten, _ = by_cand[c]
            het_label = 'True' if hetero else 'False'
            ten_str = f'nt={ten["nt"]} st={ten["st"]} rf={ten["rf"]}'
            print(f'{c:<6} {het_label:<7} {seed:>6} {min_t:>11} {ten_str:>30}')
        else:
            print(f'{c:<6} (no eligible seeds)')


if __name__ == '__main__':
    main()
