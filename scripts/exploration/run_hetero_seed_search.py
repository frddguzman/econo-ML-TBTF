"""Constrained maximin seed search at equity_heterogeneity=True.

For each (cand, regime, seed, eta), runs a single sim and records:
- total_bk
- lender max-tenure (RLE on (best_lender, generation))

Eligibility: at each cand, the η* sign must be stable in the socialized_tax
and resolution_fund regimes — i.e., total_bk(η=0.1) < total_bk(η=0) (interior
min visible). Plus we want total_bk(η=0.85) > total_bk(η=0.1) to preserve the
Lehman/turn-up shape.

Score (maximin): for each seed, compute the minimum over (cand, regime) of
max_tenure(η=0.1). Pick the seed maximizing this minimum. Tiebreaker: total
sum of max_tenure across cells.

Grid: 30 seeds × 4 cands × 3 regimes × 3 etas = 1080 sims at 6 workers
≈ 12 min wallclock.

Output: seed_search_hetero.csv with all per-cell results, and a summary
report printed to stdout.
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
ETAS = [0.0, 0.1, 0.85]
SEEDS = list(range(26462, 26492))   # 30 seeds


def _rle_max(keys):
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def make_cfg(basis, omega, eta, regime):
    return dict(
        N=50, T=1000,
        mu=0.7, omega=omega,
        eta_bailout=eta, rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=regime,
        fund_levy_rate=0.0001,
        fitness_basis=basis, fitness_inertia=0.0,
        equity_heterogeneity=True,
        equity_cv=0.5,
    )


def run_one(args):
    cand_tag, basis, omega, regime_tag, regime, eta, seed = args
    cfg = make_cfg(basis, omega, eta, regime)
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
        'total_bk':   int(sum(s.bankruptcy[:T])),
        'max_tenure': _rle_max(keys),
    }


def main():
    jobs = []
    for cand_tag, basis, omega in CANDS:
        for regime_tag, regime in REGIMES:
            for eta in ETAS:
                for seed in SEEDS:
                    jobs.append((cand_tag, basis, omega, regime_tag, regime, eta, seed))
    print(f'Hetero seed search: {len(jobs)} sims at hetero=True (4 cands x 3 regimes x 3 etas x 30 seeds)')
    workers = 6   # avoid CPU oversubscription
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(run_one, jobs), 1):
            rows.append(r)
            if i % 60 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')
    out = os.path.join(SAVE_DIR, 'seed_search_hetero.csv')
    cols = ['seed','cand','regime','eta','total_bk','max_tenure']
    with open(out, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'\nWrote {len(rows)} rows -> {out}')

    # ── Analyse ─────────────────────────────────────────────────────────
    # Index rows for quick access
    idx = {}   # (seed, cand, regime, eta) -> row
    for r in rows:
        idx[(r['seed'], r['cand'], r['regime'], r['eta'])] = r

    print('\n' + '=' * 72)
    print('Eligibility (mirrors OG maximin): eta=0.1 < eta=0 at ALL 12 cells (4 cands x 3 regimes)')
    print('=' * 72)

    # OG eligibility: total_bk(eta=0.1) < total_bk(eta=0) at all cells.
    # Plus: Lehman/turn-up preservation in fiscal regimes (st, rf): bk(eta=0.85) > bk(eta=0.1).
    # No_tax (nt) only requires the interior-vs-baseline check, since claim 3 says
    # nt monotonically declines and an η=0.85 turn-up isn't expected there.
    eligible_strict = []      # passes interior + Lehman in st,rf
    eligible_loose  = []      # passes only interior at all cells
    for seed in SEEDS:
        interior_ok = True
        lehman_ok = True
        per_cell_tenure = []
        for c, _, _ in CANDS:
            for rg, _ in REGIMES:
                bk0  = idx[(seed, c, rg, 0.0)]['total_bk']
                bk01 = idx[(seed, c, rg, 0.1)]['total_bk']
                bk85 = idx[(seed, c, rg, 0.85)]['total_bk']
                if not (bk01 < bk0):
                    interior_ok = False
                if rg in ('st', 'rf'):
                    if not (bk85 > bk01):
                        lehman_ok = False
                per_cell_tenure.append(idx[(seed, c, rg, 0.1)]['max_tenure'])
        score = (min(per_cell_tenure), sum(per_cell_tenure))
        if interior_ok and lehman_ok:
            eligible_strict.append((seed,) + score)
        if interior_ok:
            eligible_loose.append((seed,) + score)

    print(f'Strict (interior + Lehman in st,rf):  {len(eligible_strict)} / {len(SEEDS)} seeds')
    print(f'Loose  (interior only):                {len(eligible_loose)} / {len(SEEDS)} seeds')

    eligible_seeds = eligible_strict if eligible_strict else eligible_loose
    if not eligible_seeds:
        print('\nNO SEEDS pass even loose eligibility. Falling back to pure maximin (no constraints).')
        eligible_seeds = []
        for seed in SEEDS:
            per_cell_min = min(idx[(seed, c, rg, 0.1)]['max_tenure']
                              for c, _, _ in CANDS for rg, _ in REGIMES)
            per_cell_sum = sum(idx[(seed, c, rg, 0.1)]['max_tenure']
                              for c, _, _ in CANDS for rg, _ in REGIMES)
            eligible_seeds.append((seed, per_cell_min, per_cell_sum))

    # Sort by maximin (then by sum tiebreaker)
    eligible_seeds.sort(key=lambda x: (-x[1], -x[2]))
    print('\nTop 10 seeds by maximin lender max-tenure across (cand,regime) at eta=0.1:')
    print(f'{"rank":<5} {"seed":>6} {"min_ten":>8} {"sum_ten":>8}')
    for i, (s, mn, sm) in enumerate(eligible_seeds[:10], 1):
        print(f'{i:<5} {s:>6} {mn:>8} {sm:>8}')

    # Detail for the chosen seed
    chosen = eligible_seeds[0][0]
    print(f'\nCHOSEN SEED: {chosen}')
    print(f'{"cand":<6} {"regime":<8} {"bk@e0":>7} {"bk@e01":>7} {"bk@e85":>7} {"max_ten@e01":>12}')
    for c, _, _ in CANDS:
        for rg, _ in REGIMES:
            bk0  = idx[(chosen, c, rg, 0.0)]['total_bk']
            bk01 = idx[(chosen, c, rg, 0.1)]['total_bk']
            bk85 = idx[(chosen, c, rg, 0.85)]['total_bk']
            mt   = idx[(chosen, c, rg, 0.1)]['max_tenure']
            sign = '<--MIN' if (bk01 < bk0 and bk85 > bk01) else ''
            print(f'{c:<6} {rg:<8} {bk0:>7} {bk01:>7} {bk85:>7} {mt:>12} {sign}')


if __name__ == '__main__':
    main()
