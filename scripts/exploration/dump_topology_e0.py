"""Dump canonical w58 topology JSONs at η=0 (no-bailout-inflation baseline).

Mirrors dump_topology_e085.py at η=0 — the third anchor of the locked η-grid
{0, 0.10, 0.85}. SEED=26474, 3 regimes (nt, ex-post, ex-ante τ=1e-5).

At η=0 the cap denominator 1/(1−b_j·η) = 1 → no TBTF inflation (Loop 2 inactive);
the three regimes are mechanically indistinguishable at this anchor because Loop 4
is also nil (no bailouts → no tax bill). The JSONs are still produced for the three
regime labels to preserve symmetry with e01/e085 and let downstream scripts iterate
uniformly; output content is expected to be near-identical across the three at η=0.

Output: ../Simulations/topology_w58_{nt,st,t5_rf}_e0.json
Total: 3 sims at SEED=26474, η=0, ~45 sec.
"""
import os, sys, json, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SEED = 26474
ETA = 0.0
SNAPSHOTS_AT = [0, 500, 999]


def _safe(v, default=0.0):
    if v is None: return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def capture(model):
    edges, banks = [], []
    for bank in model.banks:
        bid = getattr(bank, 'id', None)
        if bid is None: continue
        bid = int(bid)
        E = round(_safe(getattr(bank, 'E', 0.0)), 3)
        clients = int(getattr(bank, 'numOfBorrowers', 0)) if hasattr(bank, 'numOfBorrowers') else 0
        banks.append({'id': bid, 'equity': E, 'clients': clients})
        lender = getattr(bank, 'lender', None)
        if lender is not None:
            l_amt = round(_safe(getattr(bank, 'l', 0.0)), 3)
            edges.append({'borrower': bid, 'lender': int(lender), 'loan': l_amt})
    return {'edges': edges, 'banks': banks}


def dump_one(label, regime, fund_levy_rate=None):
    out_path = f'../Simulations/topology_w58_{label}_e0.json'
    if os.path.exists(out_path):
        print(f'  [skip] already exists: {out_path}')
        return
    cfg = ddr.make_config(basis='equity', omega=0.58, eta=ETA, regime=regime)
    cfg['mu'] = 0.70
    cfg['gamma_capital'] = 0.10
    if fund_levy_rate is not None:
        cfg['fund_levy_rate'] = fund_levy_rate

    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Boltzmann")
    m.initialize(seed=SEED, generate_plots=False)

    snapshots = {}
    for t in range(m.config.T):
        m.forward()
        if t in SNAPSHOTS_AT:
            snapshots[str(t)] = capture(m)
    m.finish()

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, separators=(',', ':'))
    n_edges = sum(len(s['edges']) for s in snapshots.values())
    print(f'  [saved] {out_path} ({n_edges} edges across {len(snapshots)} timesteps)')


def main():
    print(f'Dumping topology JSONs at SEED={SEED}, eta={ETA}:')
    dump_one('nt', 'none')
    dump_one('st', 'socialized_tax')
    dump_one('t5_rf', 'resolution_fund', fund_levy_rate=1e-5)
    print('DONE.')


if __name__ == '__main__':
    main()
