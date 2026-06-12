"""Network topology snapshots for BA cands at the chip-focused (st, eta=0.1) cell.

Output: Simulations/topology_<cand>_ba<m>_st_e01.json (one per BA cand).
Same structure as dump_topology.py — snapshots at t=0/500/999 with edges + node attrs.

Usage:
    py -3.12 dump_topology_ba.py --m 3       # batch 1
    py -3.12 dump_topology_ba.py --m 1       # batch 2
"""
import os
os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS',  '1')

import sys
import json
import math
import argparse
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SAVE_DIR = ddr.SAVE_DIR
SNAPSHOTS_AT = [0, 500, 999]
SEED = 26474
ETA = 0.1
REGIME = 'socialized_tax'
REGIME_TAG = 'st'
ETA_TAG = 'e01'

CANDS = [
    ('bsl', 'equity',    0.50),
    ('a',   'equity',    0.53),
    ('b',   'bilateral', 0.52),
    ('w55', 'equity',    0.55),
]


def _safe(v, default=0.0):
    if v is None: return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def capture(model):
    edges = []
    banks = []
    for bank in model.banks:
        bid = getattr(bank, 'id', None)
        if bid is None: continue
        bid = int(bid)
        E = round(_safe(getattr(bank, 'E', 0.0)), 3)
        clients = int(getattr(bank, 'numOfBorrowers', 0)) if hasattr(bank, 'numOfBorrowers') else 0
        is_hub = bool(getattr(bank, 'is_hub', False)) if hasattr(bank, 'is_hub') else False
        banks.append({'id': bid, 'equity': E, 'is_hub': is_hub, 'clients': clients})
        lender = getattr(bank, 'lender', None)
        if lender is not None:
            l_amt = round(_safe(getattr(bank, 'l', 0.0)), 3)
            edges.append({'borrower': bid, 'lender': int(lender), 'loan': l_amt})
    return {'edges': edges, 'banks': banks}


def dump_one_ba(args):
    cand_tag, basis, omega, m_param = args
    cfg = ddr.make_config(basis, omega, ETA, regime=REGIME)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm("Preferential", m=m_param)
    m.initialize(seed=SEED, generate_plots=False)

    snapshots = {}
    for t in range(m.config.T):
        m.forward()
        if t in SNAPSHOTS_AT:
            snapshots[str(t)] = capture(m)
    m.finish()

    tag = f'{cand_tag}_ba{m_param}_{REGIME_TAG}_{ETA_TAG}'
    out_path = os.path.join(SAVE_DIR, f'topology_{tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, separators=(',', ':'))
    n_edges = sum(len(s['edges']) for s in snapshots.values())
    return tag, n_edges, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    m_param = args.m

    targets = [(c, b, o, m_param) for c, b, o in CANDS]
    workers = max(1, min(len(targets), 6))
    print(f'BA topology dump m={m_param} (seed={SEED}): {len(targets)} cands, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, n_edges, out in pool.map(dump_one_ba, targets):
            print(f'  {tag}: {n_edges} total edges across {len(SNAPSHOTS_AT)} snapshots')
    print('Done.')


if __name__ == '__main__':
    main()
