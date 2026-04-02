#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive parameter sweep GUI.
Pick ONE parameter to sweep, hold everything else constant,
choose which output metric(s) to plot.

Run:  python gui_sweep.py          (opens http://127.0.0.1:5001)
"""

import traceback
import os
import base64
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from flask import Flask, render_template, request, jsonify
import interbank
import interbank_lenderchange as lc

SAVE_DIR = r'C:\Users\frddg\Documents\UJI\TFG\econo-ml-TBTF\Simulations'

app = Flask(__name__)

# ── Sweepable parameters with labels, defaults, and suggested ranges ──
SWEEP_PARAMS = {
    'eta_bailout':      {'label': 'η  Bailout recovery',       'default': 0.85, 'min': 0, 'max': 1,   'step': 0.05},
    'gamma_capital':    {'label': 'γ  Capital adequacy',        'default': 0.08, 'min': 0, 'max': 0.5, 'step': 0.01},
    'alpha_collateral': {'label': 'α  Pricing recovery (eqs 4,6,8)', 'default': 0.05, 'min': 0, 'max': 0.5, 'step': 0.01},
    'beta':             {'label': 'β  Boltzmann intensity',     'default': 5,    'min': 0, 'max': 30,  'step': 1},
    'mu':               {'label': 'μ  Shock mean',              'default': 0.7,  'min': 0, 'max': 1,   'step': 0.05},
    'omega':            {'label': 'ω  Shock dispersion',        'default': 0.5,  'min': 0, 'max': 1,   'step': 0.05},
    'rho':              {'label': 'ρ  Fire sale / resolution cost', 'default': 0.3, 'min': 0, 'max': 1, 'step': 0.05},
    'phi':              {'label': 'φ  Lender screening cost',   'default': 0.025,'min': 0, 'max': 0.1, 'step': 0.005},
    'chi':              {'label': 'χ  Borrower screening cost',  'default': 0.015,'min': 0, 'max': 0.1, 'step': 0.005},
    'alfa':             {'label': 'α_BT  Bankruptcy threshold', 'default': 0.1,  'min': 0, 'max': 1,   'step': 0.01},
    'N':                {'label': 'N  Number of banks',         'default': 50,   'min': 2, 'max': 200, 'step': 1},
    'fund_levy_rate':   {'label': 'τ_fund  Resolution fund levy','default': 0.005,'min': 0, 'max': 0.05,'step': 0.001},
}

# ── Output metrics the user can choose from ──
OUTPUT_METRICS = {
    'bankruptcy':           {'label': 'Total Bankruptcies',           'color': '#dc2626'},
    'tax_induced_failures': {'label': 'Tax-Induced Failures',         'color': '#db2777'},
    'bailout_bill':         {'label': 'Total Bailout Bill',           'color': '#7c3aed'},
    'bailout_count':        {'label': 'Total Bailout Events',         'color': '#d97706'},
    'bailout_tax_total':    {'label': 'Total Bailout Tax Distributed','color': '#0891b2'},
    'equity':               {'label': 'Avg Equity (last 100 periods)','color': '#059669'},
    'interest_rate':        {'label': 'Avg Interest Rate (last 100)', 'color': '#2563eb'},
    'liquidity':            {'label': 'Avg Liquidity (last 100)',     'color': '#7c3aed'},
    'loans':                {'label': 'Avg Loans (last 100)',         'color': '#d97706'},
    'leverage':             {'label': 'Avg Leverage (last 100)',      'color': '#64748b'},
    'B':                    {'label': 'Total Bad Debt',               'color': '#dc2626'},
    'rationing':            {'label': 'Total Rationing',              'color': '#d97706'},
    'profits':              {'label': 'Total Profits',                'color': '#059669'},
    'fitness':              {'label': 'Avg Fitness (last 100)',       'color': '#059669'},
    'num_loans':            {'label': 'Avg Active Loans (last 100)',  'color': '#2563eb'},
    'active_lenders':       {'label': 'Avg Active Lenders (last 100)','color': '#0891b2'},
    'resolution_fund_balance': {'label': 'Resolution Fund Balance (last)','color': '#15803d'},
    'fund_depleted_events':    {'label': 'Fund Depletion Events',         'color': '#b91c1c'},
    'total_levy_collected':    {'label': 'Total Levy Collected',          'color': '#0369a1'},
    # Bankruptcy decomposition
    'bankruptcies_shock':      {'label': 'Bankruptcies: Shock',           'color': '#f97316'},
    'bankruptcies_rationing':  {'label': 'Bankruptcies: Rationing',       'color': '#eab308'},
    'bankruptcies_repay':      {'label': 'Bankruptcies: Repayment',       'color': '#8b5cf6'},
    'bankruptcies_contagion':  {'label': 'Bankruptcies: Contagion',       'color': '#ef4444'},
    'bankruptcies_fiscal':     {'label': 'Bankruptcies: Fiscal',          'color': '#ec4899'},
    'fire_sale_survivors':     {'label': 'Fire Sale Survivors (Zombies)', 'color': '#6b7280'},
}

# Metrics where we take sum() vs mean-of-last-100
SUM_METRICS = {'bankruptcy', 'tax_induced_failures', 'bailout_bill', 'bailout_count',
               'bailout_tax_total', 'B', 'rationing', 'profits',
               'fund_depleted_events', 'total_levy_collected',
               'bankruptcies_shock', 'bankruptcies_rationing', 'bankruptcies_repay',
               'bankruptcies_contagion', 'bankruptcies_fiscal', 'fire_sale_survivors'}

# Metrics where we take the last value (end-of-simulation state)
LAST_METRICS = {'resolution_fund_balance'}


def extract_metric(stats, name, T):
    """Extract a single scalar from a statistics array."""
    arr = getattr(stats, name, None)
    if arr is None:
        return 0.0
    if name in LAST_METRICS:
        return float(arr[T - 1]) if T > 0 else 0.0
    elif name in SUM_METRICS:
        return float(np.nansum(arr[:T]))
    else:
        tail = max(0, T - 100)
        return float(np.nanmean(arr[tail:T]))


def _run_single(args):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    cfg_base, sweep_key, val, seed, algorithm, lc_p, lc_m, metrics = args
    model = interbank.Model()
    model.test = True
    cfg = dict(cfg_base)
    cfg[sweep_key] = val
    if 'N' in cfg:
        cfg['N'] = int(cfg['N'])
    if 'T' in cfg:
        cfg['T'] = int(cfg['T'])
    model.configure(**cfg)
    model.config.lender_change = lc.determine_algorithm(algorithm, p=lc_p, m=lc_m)
    model.initialize(seed=seed, generate_plots=False)
    model.simulate_full()
    model.finish()
    T = model.t
    row = {}
    for m in metrics:
        row[m] = extract_metric(model.statistics, m, T)
    return (val, seed, row)


@app.route('/')
def index():
    return render_template('index_sweep.html',
                           sweep_params=SWEEP_PARAMS,
                           output_metrics=OUTPUT_METRICS)


@app.route('/api/sweep', methods=['POST'])
def api_sweep():
    try:
        body = request.get_json()

        sweep_key = body['sweep_param']
        sweep_values = body['sweep_values']
        metrics = body['metrics']
        fixed = body.get('config', body.get('fixed', {}))
        mc_enabled = body.get('mc_enabled', False)
        mc_seeds = int(body.get('mc_seeds', 20))
        workers = int(body.get('workers', 6))

        def generate():
            import json

            cfg_base = dict(fixed)
            base_seed = int(cfg_base.pop('seed', 26462))
            algorithm = cfg_base.pop('algorithm', 'Boltzmann')
            lc_p = float(cfg_base.pop('lc_p', 0.5))
            lc_m = int(cfg_base.pop('lc_m', 1))

            # Build job list
            jobs = []
            if mc_enabled:
                for val in sweep_values:
                    for s in range(mc_seeds):
                        jobs.append((cfg_base, sweep_key, val, base_seed + s,
                                     algorithm, lc_p, lc_m, metrics))
            else:
                for val in sweep_values:
                    jobs.append((cfg_base, sweep_key, val, base_seed,
                                 algorithm, lc_p, lc_m, metrics))

            total = len(jobs)
            actual_workers = min(workers, total)
            done_count = 0

            # Run all jobs in parallel
            all_results = []  # list of (val, seed, row)
            with ProcessPoolExecutor(max_workers=actual_workers) as pool:
                futures = {pool.submit(_run_single, job): job for job in jobs}
                for future in as_completed(futures):
                    val, seed, row = future.result()
                    all_results.append((val, seed, row))
                    done_count += 1
                    status_msg = f'{sweep_key}={val}'
                    if mc_enabled:
                        status_msg += f' seed={seed}'
                    print(f"  [{done_count}/{total}] {status_msg} done")
                    yield json.dumps({
                        'type': 'progress',
                        'done': done_count,
                        'total': total,
                        'status': status_msg
                    }) + '\n'

            if not mc_enabled:
                # Reconstruct results in sweep_values order
                result_map = {val: row for val, seed, row in all_results}
                run_results = [result_map[val] for val in sweep_values]

                yield json.dumps({
                    'type': 'result',
                    'mc': False,
                    'sweep_values': sweep_values,
                    'values': run_results
                }) + '\n'
            else:
                # Group by sweep value, aggregate
                by_val = defaultdict(list)
                for val, seed, row in all_results:
                    by_val[val].append(row)

                agg_results = []
                for val in sweep_values:
                    seed_rows = by_val[val]
                    agg = {}
                    for m in metrics:
                        vals = [r[m] for r in seed_rows]
                        agg[m] = {
                            'mean': float(np.mean(vals)),
                            'std':  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                            'min':  float(np.min(vals)),
                            'max':  float(np.max(vals)),
                        }
                    agg_results.append(agg)

                yield json.dumps({
                    'type': 'result',
                    'mc': True,
                    'mc_seeds': mc_seeds,
                    'sweep_values': sweep_values,
                    'values': agg_results
                }) + '\n'

        return app.response_class(generate(), mimetype='text/plain')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save', methods=['POST'])
def api_save():
    try:
        body = request.get_json()
        name = body['name']
        png_b64 = body['png']       # base64-encoded PNG from Plotly
        csv_text = body['csv']

        os.makedirs(SAVE_DIR, exist_ok=True)

        png_path = os.path.join(SAVE_DIR, name + '.png')
        csv_path = os.path.join(SAVE_DIR, name + '.csv')

        with open(png_path, 'wb') as f:
            f.write(base64.b64decode(png_b64))
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_text)

        return jsonify({'ok': True, 'png': png_path, 'csv': csv_path})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Sweep GUI running at http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=False)
