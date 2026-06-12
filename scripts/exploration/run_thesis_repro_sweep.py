"""Thesis-reproduction sweep: produce three CSVs in Simulations/ that let the
dashboard reproduce Claims 1, 2, 3 at the chosen candidate baseline.

Default candidate is the thesis baseline (bsl: equity, ω=0.50). Pass --cand a
or --cand b to sweep the other candidate baselines:
    bsl: equity   λ=0  ω=0.50  (matches thesis figure 3.10)
    a:   equity   λ=0  ω=0.53  (Cand A — hub-formation candidate)
    b:   bilateral λ=0 ω=0.52  (Cand B — hub-formation candidate)

Outputs (suffixed with candidate tag for non-baseline runs):
    thesis_lehman[_a|_b].csv   η ∈ {0..0.9 step 0.1} × {none, socialized_tax, resolution_fund}
                               × seeds 26462–26466,  ρ=0.4   (Claim 3 + Lehman zone)
    thesis_rho[_a|_b].csv      ρ ∈ {0..0.8 step 0.1} × η=0, no-tax × seeds 26462–26466
                               (Claim 1 zombie activation + Claim 2 non-monotonic peak)
    thesis_rho01[_a|_b].csv    η ∈ {0,0.1,0.2,0.3,0.5,0.7,0.9} × {none, resolution_fund}
                               × seeds 26462–26466, ρ=0.1  (Claim 3 secondary)

Run from the econo-ML-TBTF-clean directory:
    python run_thesis_repro_sweep.py            # baseline (bsl)
    python run_thesis_repro_sweep.py --cand a   # cand A (equity ω=0.53)
    python run_thesis_repro_sweep.py --cand b   # cand B (bilateral ω=0.52)
"""
import os
import sys
import math
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Simulations')
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = [26462, 26463, 26464, 26465, 26466]
WORKERS = 6   # 6 physical cores; oversubscribing (e.g. 11) thrashes the CPU

# Mirror dump_dashboard_runs.CAND_DEFS — keeps the parameter set authoritative.
CAND_PARAMS = {
    'bsl': {'basis': 'equity',    'omega': 0.50, 'inertia': 0.0},
    'a':   {'basis': 'equity',    'omega': 0.53, 'inertia': 0.0},
    'b':   {'basis': 'bilateral', 'omega': 0.52, 'inertia': 0.0},
    'w55': {'basis': 'equity',    'omega': 0.55, 'inertia': 0.0},
}

# ── Common config — matches dump_dashboard_runs.make_config (the new candidate baseline) ──
# fund_levy_rate=0.0001 matches gui_zombie.py default (the authoritative one); the
# interbank.py default of 0.005 is 50× too high — at that level the levy alone
# bankrupts every bank within ~20 periods, breaking the resolution_fund mechanism.
def make_config(eta, rho=0.4, regime='socialized_tax',
                omega=0.50, basis='equity', inertia=0.0):
    return dict(
        N=50, T=1000,
        mu=0.7,
        omega=omega,
        eta_bailout=eta,
        rho=rho,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=regime,
        fund_levy_rate=0.0001,
        fitness_basis=basis,
        fitness_inertia=inertia,
    )


def _sum_int(stats, name, T):
    arr = getattr(stats, name, None)
    if arr is None:
        return 0
    return int(np.nansum(arr[:T]))


def run_one(args):
    """One simulation; returns dict of metric totals.

    args is a 7-tuple OR 8-tuple. The optional 8th element is m_param: when
    set, the lender_change algorithm is Preferential(m=m_param) instead of
    Boltzmann. This produces BA-topology runs whose output sits alongside the
    Boltzmann thesis CSVs (different file names — see main()).
    """
    if len(args) == 8:
        seed, eta, rho, regime, basis, omega, inertia, m_param = args
    else:
        seed, eta, rho, regime, basis, omega, inertia = args
        m_param = None
    cfg = make_config(eta=eta, rho=rho, regime=regime,
                      basis=basis, omega=omega, inertia=inertia)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    if m_param is not None:
        m.config.lender_change = lc.determine_algorithm('Preferential', m=m_param)
    else:
        m.config.lender_change = lc.determine_algorithm('Boltzmann')
    m.initialize(seed=seed, generate_plots=False)
    m.simulate_full()
    m.finish()
    T = m.t
    s = m.statistics
    return {
        'seed':           int(seed),
        'fiscal_regime':  regime,
        'rho':            float(rho),
        'eta':            float(eta),
        'omega':          float(cfg['omega']),
        'basis':          basis,
        'inertia':        float(inertia),
        'total_bk':       _sum_int(s, 'bankruptcy', T),
        'shock':          _sum_int(s, 'bankruptcies_shock', T),
        'rationing':      _sum_int(s, 'bankruptcies_rationing', T),
        'repay':          _sum_int(s, 'bankruptcies_repay', T),
        'contagion':      _sum_int(s, 'bankruptcies_contagion', T),
        'fiscal_deaths':  _sum_int(s, 'bankruptcies_fiscal', T),
        'zombies':        _sum_int(s, 'fire_sale_survivors', T),
    }


def write_csv(rows, path, columns):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(columns) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in columns) + '\n')
    print(f'  wrote {len(rows):4d} rows -> {path}')


def run_phase(name, jobs, out_path, columns):
    print(f'\n[{name}]  {len(jobs)} sims')
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} done')
    write_csv(rows, out_path, columns)


COLUMNS = [
    'seed', 'fiscal_regime', 'rho', 'eta', 'omega', 'basis', 'inertia',
    'total_bk', 'shock', 'rationing', 'repay', 'contagion', 'fiscal_deaths', 'zombies',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cand', choices=['bsl', 'a', 'b', 'w55'], default='bsl',
                        help='Candidate baseline (default bsl = thesis baseline)')
    parser.add_argument('--m', type=int, default=None, choices=[1, 2, 3, 5],
                        help='Preferential m param. If set, sweep uses BA topology '
                             'instead of Boltzmann; output gets _ba<m> suffix.')
    args = parser.parse_args()
    cand = args.cand
    p = CAND_PARAMS[cand]
    m_param = args.m

    # Output suffix: _<cand> for non-baseline + _ba<m> for BA runs
    cand_suffix = '' if cand == 'bsl' else f'_{cand}'
    ba_suffix   = '' if m_param is None else f'_ba{m_param}'
    suffix = cand_suffix + ba_suffix
    algo_label = f'Preferential (BA m={m_param})' if m_param is not None else 'Boltzmann'
    print(f'Candidate {cand}: basis={p["basis"]}  omega={p["omega"]}  '
          f'lambda={p["inertia"]}  algo={algo_label}')

    # Helper: append m_param to job tuple if BA mode
    def _job(s, eta, rho, regime):
        base = (s, eta, rho, regime, p['basis'], p['omega'], p['inertia'])
        return base + (m_param,) if m_param is not None else base

    # Lehman: η × 3 regimes × seeds, ρ=0.4
    etas_full = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    regimes_full = ['none', 'socialized_tax', 'resolution_fund']
    lehman_jobs = [
        _job(s, eta, 0.4, regime)
        for s in SEEDS for eta in etas_full for regime in regimes_full
    ]
    run_phase(
        'thesis_lehman' + suffix, lehman_jobs,
        os.path.join(SAVE_DIR, f'thesis_lehman{suffix}.csv'),
        COLUMNS,
    )

    # ρ-sweep: ρ ∈ {0..0.8} × η=0 × no-tax × seeds
    rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rho_jobs = [
        _job(s, 0.0, rho, 'none')
        for s in SEEDS for rho in rhos
    ]
    run_phase(
        'thesis_rho' + suffix, rho_jobs,
        os.path.join(SAVE_DIR, f'thesis_rho{suffix}.csv'),
        COLUMNS,
    )

    # ρ=0.1 secondary: η × {none, resolution_fund} × seeds
    etas_rho01 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    rho01_jobs = [
        _job(s, eta, 0.1, regime)
        for s in SEEDS for eta in etas_rho01
        for regime in ('none', 'resolution_fund')
    ]
    run_phase(
        'thesis_rho01' + suffix, rho01_jobs,
        os.path.join(SAVE_DIR, f'thesis_rho01{suffix}.csv'),
        COLUMNS,
    )

    print('\nDone.  Re-run gen_dashboard.py to embed.')


if __name__ == '__main__':
    main()
