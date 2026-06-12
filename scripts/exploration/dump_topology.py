"""Capture network-topology snapshots at three timepoints (t=0, 500, 1000)
for the chip-focused (regime, eta) cell, for each of the 4 candidates. Output:
    Simulations/topology_<cand>_st_e01.json

Each JSON has snapshots keyed by period, each containing:
- edges: [{borrower, lender, loan}]  — the borrower→lender edge list
- banks: [{id, equity, is_hub, clients}]  — node attributes

Used by gen_dashboard.py to render Brini Fig. 3 analogue (3 side-by-side node-link
diagrams per cand).

Run: py -3.12 dump_topology.py  (~90 sec for 4 sims at T=1000, parallel)
"""
import os
import sys
import json
import math
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interbank
import interbank_lenderchange as lc
import dump_dashboard_runs as ddr

SAVE_DIR = ddr.SAVE_DIR
SNAPSHOTS_AT = [0, 500, 999]   # captured AFTER each forward(), so end-of-period state
SEED = 26474
ETA = 0.1
REGIME = 'socialized_tax'
REGIME_TAG = 'st'
ETA_TAG = 'e01'

# 4 cands × 1 regime × 1 eta = 4 sims
TARGETS = [
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
        if bid is None:
            continue
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


def dump_one(args):
    cand_tag, basis, omega = args
    cfg = ddr.make_config(basis, omega, ETA, regime=REGIME)
    m = interbank.Model()
    m.test = True
    m.configure(**cfg)
    m.config.lender_change = lc.determine_algorithm('Boltzmann')
    m.initialize(seed=SEED, generate_plots=False)

    snapshots = {}
    for t in range(m.config.T):
        m.forward()
        if t in SNAPSHOTS_AT:
            snapshots[str(t)] = capture(m)
    m.finish()

    tag = f'{cand_tag}_{REGIME_TAG}_{ETA_TAG}'
    out_path = os.path.join(SAVE_DIR, f'topology_{tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, separators=(',', ':'))
    n_edges = sum(len(s['edges']) for s in snapshots.values())
    return tag, n_edges, out_path


def main():
    workers = max(1, min(len(TARGETS), 6))   # 6 physical cores cap
    print(f'topology dump (seed={SEED}, regime={REGIME}, eta={ETA}): '
          f'{len(TARGETS)} sims, {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for tag, n_edges, out in pool.map(dump_one, TARGETS):
            print(f'  {tag}: {n_edges} total edges across {len(SNAPSHOTS_AT)} snapshots -> {out}')
    print('\nDone.')


if __name__ == '__main__':
    main()
