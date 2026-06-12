"""Preferential m=3 screening across 4 candidate baselines.

For each cand ∈ {bsl, a, b, w55} runs TWO sims at η=0.1, st, seed 26474:
  baseline: Boltzmann (current model)
  pref:     Preferential m=3 (BA topology + cash-boost guru)

Captures hub max_tenure, in-degree distribution, guru tracking, and writes a
topology JSON at t=999 per (cand, algorithm) for visual inspection.

Outputs:
- stdout: A/B comparison table per cand
- pref_screening_<cand>_<algo>.json × 8 in project root
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc

SEED = 26474
ETA = 0.1
REGIME = 'socialized_tax'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CANDS = [
    ('bsl', 'equity',    0.50),
    ('a',   'equity',    0.53),
    ('b',   'bilateral', 0.52),
    ('w55', 'equity',    0.55),
]

ALGORITHMS = [
    ('boltz',  'Boltzmann',    None),
    ('pref1',  'Preferential', 1),    # tree: every node has 1 outgoing edge → super-hub
    ('pref2',  'Preferential', 2),    # minimal scale-free
    ('pref3',  'Preferential', 3),    # classic BA
]


def make_cfg(basis, omega):
    return dict(
        N=50, T=1000,
        mu=0.7, omega=omega,
        eta_bailout=ETA, rho=0.4,
        gamma_capital=0.10, alpha_collateral=0.05,
        beta=5, alfa=0.1,
        fiscal_regime=REGIME,
        fund_levy_rate=0.0001,
        fitness_basis=basis, fitness_inertia=0.0,
        equity_heterogeneity=False, equity_cv=0.5,
    )


def _rle_max(keys):
    if not keys: return 0
    runs = []; cur = keys[0]; start = 0
    for i in range(1, len(keys)):
        if keys[i] != cur:
            runs.append(i - start); cur = keys[i]; start = i
    runs.append(len(keys) - start)
    return max(runs)


def _topology_snapshot(model):
    """Capture edge list + node attributes at the current model state."""
    nodes = []
    edges = []
    in_deg = {}
    for b in model.banks:
        if b.failed: continue
        in_deg[b.id] = 0
    for b in model.banks:
        if b.failed: continue
        if b.lender is not None and b.lender < len(model.banks):
            edges.append({
                'borrower': int(b.id),
                'lender':   int(b.lender),
                'loan':     round(float(b.l) if hasattr(b, 'l') else 0.0, 3),
            })
            in_deg[b.lender] = in_deg.get(b.lender, 0) + 1
    for b in model.banks:
        if b.failed: continue
        nodes.append({
            'id':      int(b.id),
            'equity':  round(float(b.E), 3),
            'fitness': round(float(b.mu), 4) if b.mu is not None else 0,
            'in_deg':  int(in_deg.get(b.id, 0)),
        })
    return {'nodes': nodes, 'edges': edges}


def run_one(cand_tag, basis, omega, algo_tag, algo_name, m_param):
    cfg = make_cfg(basis, omega)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    if m_param is not None:
        m.config.lender_change = lc.determine_algorithm(algo_name, m=m_param)
    else:
        m.config.lender_change = lc.determine_algorithm(algo_name)
    m.initialize(seed=SEED, generate_plots=False)

    initial_guru = getattr(m.config.lender_change, 'guru', None)
    initial_C_max = max(b.C for b in m.banks)
    initial_E_max = max(b.E for b in m.banks)

    snapshot_at_999 = None
    initial_guru_failed_at = None

    for t in range(m.config.T):
        m.forward()
        if t == 999:
            snapshot_at_999 = _topology_snapshot(m)
        if (initial_guru is not None and initial_guru < len(m.banks)
                and m.banks[initial_guru].failed
                and initial_guru_failed_at is None):
            initial_guru_failed_at = t

    m.finish()
    T = m.t
    s = m.statistics
    bl = list(s.best_lender[:T]); bg = list(s.best_lender_generation[:T])
    keys = [(b, g) for b, g in zip(bl, bg) if b >= 0]
    deg = list(s.best_lender_clients[:T])
    deg_pos = [d for d in deg if d is not None and d >= 0]
    hub_trace = [int(x) for x in bl if x >= 0]

    time_at_initial_guru = (sum(1 for h in hub_trace if h == initial_guru)
                            if initial_guru is not None else 0)
    unique_hubs = len(set(hub_trace))

    return {
        'cand_tag':          cand_tag,
        'algo_tag':          algo_tag,
        'algo_name':         algo_name,
        'm_param':           m_param,
        'initial_guru':      initial_guru,
        'initial_C_max':     initial_C_max,
        'initial_E_max':     initial_E_max,
        'total_bk':          int(sum(s.bankruptcy[:T])),
        'max_tenure':        _rle_max(keys),
        'mean_deg':          statistics.mean(deg_pos) if deg_pos else 0,
        'max_deg':           max(deg_pos) if deg_pos else 0,
        'p95_deg':           (statistics.quantiles(deg_pos, n=20)[18]
                              if len(deg_pos) >= 20 else max(deg_pos, default=0)),
        'guru_failed_at':    initial_guru_failed_at,
        'time_at_initial_guru': time_at_initial_guru,
        'unique_hubs':       unique_hubs,
        'snapshot_999':      snapshot_at_999,
    }


def main():
    print(f'Preferential m=3 screening across 4 cands (eta=0.1, st, seed {SEED})')
    print('=' * 110)
    print(f'{"cand":<5} {"algo":<7} {"tot_bk":>7} {"max_t":>6} {"mean_d":>7} {"max_d":>6} {"p95":>4} '
          f'{"guru":>5} {"C_max":>7} {"hub_at_guru":>11} {"unique_hubs":>11}')
    print('-' * 110)

    results = []
    for cand_tag, basis, omega in CANDS:
        for algo_tag, algo_name, m_param in ALGORITHMS:
            r = run_one(cand_tag, basis, omega, algo_tag, algo_name, m_param)
            results.append(r)
            print(f'{cand_tag:<5} {algo_tag:<7} {r["total_bk"]:>7} {r["max_tenure"]:>6} '
                  f'{r["mean_deg"]:>7.2f} {r["max_deg"]:>6} {int(r["p95_deg"]):>4} '
                  f'{str(r["initial_guru"]):>5} {r["initial_C_max"]:>7.1f} '
                  f'{r["time_at_initial_guru"]:>11} {r["unique_hubs"]:>11}')

            # Save topology snapshot
            if r['snapshot_999'] is not None:
                fname = f'pref_screening_{cand_tag}_{algo_tag}.json'
                fpath = os.path.join(PROJECT_ROOT, fname)
                with open(fpath, 'w', encoding='utf-8') as f:
                    json.dump(r['snapshot_999'], f, separators=(',', ':'))

    print('\n' + '=' * 110)
    print('Pass-criteria check:')
    pref_results = [r for r in results if r['algo_tag'] == 'pref3']
    n_pass_deg = sum(1 for r in pref_results if r['max_deg'] >= 15)
    n_pass_ten = sum(1 for r in pref_results if r['max_tenure'] >= 8)
    n_with_guru = sum(1 for r in pref_results if r['initial_guru'] is not None)
    print(f'  cands with max_deg >= 15:    {n_pass_deg}/4')
    print(f'  cands with max_tenure >= 8:  {n_pass_ten}/4')
    print(f'  cands with identifiable guru: {n_with_guru}/4')
    if n_pass_deg >= 3 and n_pass_ten >= 3 and n_with_guru >= 3:
        print('\n  VERDICT: PASS — green-light full dashboard integration.')
    else:
        print('\n  VERDICT: FAIL — investigate before committing.')

    print(f'\nTopology JSONs written to {PROJECT_ROOT}/pref_screening_<cand>_<algo>.json')


if __name__ == '__main__':
    main()
