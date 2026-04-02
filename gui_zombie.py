#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zombie Lending Channel Dashboard.
Interactive GUI demonstrating how bailout expectations create zombie lending
dynamics in a TBTF interbank network model.

Run:  python gui_zombie.py          (opens http://127.0.0.1:5003)
"""

import matplotlib
matplotlib.use('Agg')

import os
import io
import base64
import math
import traceback
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from flask import Flask, render_template, request, jsonify

import interbank
import interbank_lenderchange as lc

app = Flask(__name__)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Simulations')

# ── Valid lender-change algorithms ──
VALID_ALGORITHMS = [
    'Boltzmann', 'InitialStability', 'Preferential',
    'RestrictedMarket', 'ShockedMarket', 'ShockedMarket2',
    'ShockedMarket3', 'SmallWorld',
]

# ── Sweepable parameters ──
SWEEP_PARAMS = {
    'eta_bailout':       {'label': '\u03b7  Bailout recovery',         'default': 0.85, 'min': 0,    'max': 1,    'step': 0.1},
    'rho':               {'label': '\u03c1  Fire sale / resolution cost', 'default': 0.4,  'min': 0,    'max': 1,    'step': 0.1},
    'gamma_capital':     {'label': '\u03b3  Capital adequacy',         'default': 0.10, 'min': 0.01, 'max': 0.5,  'step': 0.01},
    'alpha_collateral':  {'label': '\u03b1  Pricing recovery',         'default': 0.05, 'min': 0.01, 'max': 0.5,  'step': 0.01},
    'fund_levy_rate':    {'label': '\u03c4  Fund levy rate',           'default': 0.0001, 'min': 0,  'max': 0.01, 'step': 0.0001},
    'beta':              {'label': '\u03b2  Boltzmann intensity',      'default': 5,    'min': 0,    'max': 20,   'step': 1},
    'mu':                {'label': '\u03bc  Shock mean',               'default': 0.7,  'min': 0.1,  'max': 1.0,  'step': 0.05},
    'omega':             {'label': '\u03c9  Shock dispersion',         'default': 0.5,  'min': 0.1,  'max': 1.0,  'step': 0.05},
    'alfa':              {'label': '\u03b1_BT Bankruptcy threshold',   'default': 0.1,  'min': 0.01, 'max': 1.0,  'step': 0.01},
    'N':                 {'label': 'N  Number of banks',               'default': 50,   'min': 5,    'max': 200,  'step': 5},
}

# ── Output metrics ──
OUTPUT_METRICS = {
    'bankruptcy':              {'label': 'Total Bankruptcies',           'color': '#dc2626'},
    'bankruptcies_shock':      {'label': 'Bankruptcies: Shock',          'color': '#f97316'},
    'bankruptcies_rationing':  {'label': 'Bankruptcies: Rationing',      'color': '#eab308'},
    'bankruptcies_repay':      {'label': 'Bankruptcies: Repayment',      'color': '#8b5cf6'},
    'bankruptcies_contagion':  {'label': 'Bankruptcies: Contagion',      'color': '#ef4444'},
    'bankruptcies_fiscal':     {'label': 'Bankruptcies: Fiscal',         'color': '#ec4899'},
    'fire_sale_survivors':     {'label': 'Fire Sale Survivors (Zombies)', 'color': '#6b7280'},
    'tax_induced_failures':    {'label': 'Tax-Induced Failures',         'color': '#db2777'},
    'bailout_bill':            {'label': 'Total Bailout Bill',           'color': '#7c3aed'},
    'bailout_count':           {'label': 'Total Bailout Events',         'color': '#d97706'},
    'equity':                  {'label': 'Avg Equity (last 100)',        'color': '#059669'},
    'interest_rate':           {'label': 'Avg Interest Rate (last 100)', 'color': '#2563eb'},
    'liquidity':               {'label': 'Avg Liquidity (last 100)',     'color': '#7c3aed'},
    'loans':                   {'label': 'Avg Loans (last 100)',         'color': '#d97706'},
    'B':                       {'label': 'Total Bad Debt',               'color': '#dc2626'},
    'rationing':               {'label': 'Total Rationing',              'color': '#d97706'},
    'leverage':                {'label': 'Avg Leverage (last 100)',      'color': '#64748b'},
    'resolution_fund_balance': {'label': 'Fund Balance (last)',          'color': '#15803d'},
    'fund_depleted_events':    {'label': 'Fund Depletion Events',        'color': '#b91c1c'},
    'total_levy_collected':    {'label': 'Total Levy Collected',         'color': '#0369a1'},
}

# Metrics where we take sum() vs mean-of-last-100
SUM_METRICS = {
    'bankruptcy', 'tax_induced_failures', 'bailout_bill', 'bailout_count',
    'B', 'rationing', 'fund_depleted_events', 'total_levy_collected',
    'bankruptcies_shock', 'bankruptcies_rationing', 'bankruptcies_repay',
    'bankruptcies_contagion', 'bankruptcies_fiscal', 'fire_sale_survivors',
}

# Metrics where we take the last value
LAST_METRICS = {'resolution_fund_balance'}

# All per-period statistics to extract for single simulation
STAT_NAMES = [
    'bankruptcy', 'bankruptcies_shock', 'bankruptcies_rationing',
    'bankruptcies_repay', 'bankruptcies_contagion', 'bankruptcies_fiscal',
    'fire_sale_survivors',
    'equity', 'interest_rate', 'liquidity', 'loans', 'B', 'rationing',
    'leverage', 'profits',
    'bailout_bill', 'bailout_count', 'tax_induced_failures',
    'resolution_fund_balance', 'fund_depleted_events', 'total_levy_collected',
]


# ── Helpers ──

def safe_list(arr, length):
    """Convert a numpy array slice to a JSON-safe Python list (NaN/Inf -> None)."""
    values = arr[:length].tolist()
    return [
        None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
        for v in values
    ]


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


def _build_config(params):
    """Build config dict from request parameters."""
    config_params = {}

    # Integer params
    for k in ('N', 'T'):
        if k in params:
            config_params[k] = int(params[k])

    # Float params
    float_keys = [
        'rho', 'beta', 'alfa', 'mu', 'omega',
        'gamma_capital', 'eta_bailout', 'alpha_collateral',
        'phi', 'chi',
        'fund_levy_rate', 'fund_initial_balance',
    ]
    for k in float_keys:
        if k in params:
            config_params[k] = float(params[k])

    # String params
    if 'fiscal_regime' in params:
        config_params['fiscal_regime'] = str(params['fiscal_regime'])

    return config_params


def _create_and_run_model(config_params, seed, algorithm, lc_p, lc_m):
    """Create, configure, and run a model. Returns the model object."""
    model = interbank.Model()
    model.test = True
    model.configure(**config_params)
    model.config.lender_change = lc.determine_algorithm(algorithm, p=lc_p, m=lc_m)
    model.initialize(seed=seed, generate_plots=False)
    model.simulate_full()
    model.finish()
    return model


def build_network_image(model):
    """Render the final bank->lender network as a base64-encoded PNG.

    Clean style: white background, health-gradient node coloring,
    asset-proportional node sizes, directional edges.
    """
    G = nx.DiGraph()
    alive_banks = [b for b in model.banks if not b.failed]
    failed_ids = {b.id for b in model.banks if b.failed}

    for bank in model.banks:
        G.add_node(bank.id)
        if bank.lender is not None and bank.lender < len(model.banks):
            G.add_edge(bank.id, bank.lender)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, 'No banks remaining', ha='center', va='center',
                fontsize=14, color='#666')
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, len(G.nodes) ** 0.5))

        # Color nodes by health: green (healthy) -> red (near failure) -> gray (failed)
        alfa_threshold = getattr(model.config, 'alfa', 0.1)
        max_e = max((b.E for b in model.banks if not b.failed), default=1.0)
        max_e = max(max_e, 1e-9)

        # Green-to-red colormap for health
        health_cmap = mcolors.LinearSegmentedColormap.from_list(
            'health', ['#dc2626', '#f59e0b', '#16a34a'])

        colors = []
        for node in G.nodes():
            b = model.banks[node]
            if b.failed:
                colors.append('#d1d5db')  # gray for failed
            else:
                # Health ratio: 0 = near threshold, 1 = max equity
                health = min(1.0, max(0.0, (b.E / max_e)))
                colors.append(health_cmap(health))

        # Size nodes by total assets
        max_a = max((model.banks[n].A for n in G.nodes()
                     if not model.banks[n].failed), default=1.0)
        max_a = max(max_a, 1e-9)
        sizes = []
        for node in G.nodes():
            b = model.banks[node]
            if b.failed:
                sizes.append(200)
            else:
                sizes.append(200 + 800 * (b.A / max_a))

        # Edge colors: red for edges involving failed banks
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            if u in failed_ids or v in failed_ids:
                edge_colors.append('#ef4444')
                edge_widths.append(1.5)
            else:
                edge_colors.append('#d1d5db')
                edge_widths.append(0.8)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                               edgecolors='#9ca3af', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                               arrowstyle='->', arrowsize=12,
                               edge_color=edge_colors, width=edge_widths,
                               alpha=0.6, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color='#1f2937',
                                font_weight='bold')

    ax.set_title('Bank Network (borrower \u2192 lender)\nNode size ~ total assets, color ~ equity health',
                 fontsize=11, color='#374151', pad=12)
    ax.axis('off')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#ffffff', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def collect_bank_snapshot(model):
    """Per-bank snapshot at the current period for network tooltip data."""
    banks_data = []
    max_a_lagged = getattr(model, 'max_A_lagged', 1.0) or 1.0
    max_e = getattr(model, 'maxE', 1.0) or 1.0

    for bank in model.banks:
        A = getattr(bank, 'A', 0)
        A_lag = getattr(bank, 'A_lagged', A)
        E = getattr(bank, 'E', 0)
        p_j = 1 - (E / max_e) if max_e > 0 else 1.0
        b_j = A_lag / max_a_lagged if max_a_lagged > 0 else 0
        fitness = E / max_e if max_e > 0 else 0

        # Count borrowers: banks whose lender is this bank
        n_borrowers = sum(
            1 for b in model.banks
            if not b.failed and b.id != bank.id
            and b.get_lender() is not None and b.get_lender().id == bank.id
        )

        banks_data.append({
            'id': bank.id,
            'E': round(float(E), 4),
            'A': round(float(A), 4),
            'A_lagged': round(float(A_lag), 4),
            'failed': bool(bank.failed),
            'b_j': round(float(b_j), 4),
            'p_j': round(float(p_j), 4),
            'fitness': round(float(fitness), 4),
            'lender': bank.lender,
            'num_borrowers': n_borrowers,
        })

    return banks_data


# ── Module-level function for ProcessPoolExecutor (must be picklable) ──

def _run_single(args):
    """Run one simulation and extract requested metrics."""
    cfg_dict, seed, algorithm, lc_p, lc_m, metrics = args
    cfg = dict(cfg_dict)
    if 'N' in cfg:
        cfg['N'] = int(cfg['N'])
    if 'T' in cfg:
        cfg['T'] = int(cfg['T'])
    model = _create_and_run_model(cfg, seed, algorithm, lc_p, lc_m)
    T = model.t
    row = {}
    for m in metrics:
        row[m] = extract_metric(model.statistics, m, T)
    return row


def _run_single_sweep(args):
    """Run one simulation for a sweep job. Returns (sweep_val, seed, metrics_row)."""
    cfg_base, sweep_key, val, seed, algorithm, lc_p, lc_m, metrics = args
    cfg = dict(cfg_base)
    cfg[sweep_key] = val
    if 'N' in cfg:
        cfg['N'] = int(cfg['N'])
    if 'T' in cfg:
        cfg['T'] = int(cfg['T'])
    model = _create_and_run_model(cfg, seed, algorithm, lc_p, lc_m)
    T = model.t
    row = {}
    for m in metrics:
        row[m] = extract_metric(model.statistics, m, T)
    return (val, seed, row)


# ── Routes ──

@app.route('/')
def index():
    return render_template('index_zombie.html',
                           sweep_params=SWEEP_PARAMS,
                           output_metrics=OUTPUT_METRICS)


@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """Single simulation with full decomposition and network graph."""
    try:
        params = request.get_json(force=True)

        # Extract algorithm params
        algorithm = params.get('algorithm', 'Boltzmann')
        if algorithm not in VALID_ALGORITHMS:
            algorithm = 'Boltzmann'
        lc_p = float(params.get('lc_p', 0.5))
        lc_m = int(params.get('lc_m', 1))
        seed = int(params.get('seed', 26462))

        config_params = _build_config(params)

        # Run simulation
        model = interbank.Model()
        model.test = True
        model.configure(**config_params)
        model.config.lender_change = lc.determine_algorithm(algorithm, p=lc_p, m=lc_m)
        model.initialize(seed=seed, generate_plots=False)
        model.simulate_full()
        model.finish()

        t = model.t

        # Build result with time-series data
        result = {'time': list(range(t))}

        for name in STAT_NAMES:
            arr = getattr(model.statistics, name, None)
            if arr is not None:
                result[name] = safe_list(arr, t)

        # Network graph image
        result['network'] = build_network_image(model)

        # Bank snapshot at final period
        result['banks'] = collect_bank_snapshot(model)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/sweep', methods=['POST'])
def api_sweep():
    """Parameter sweep with decomposition metrics (streaming JSON)."""
    try:
        body = request.get_json()

        sweep_key = body['sweep_param']
        metrics = body['metrics']
        workers = int(body.get('workers', 6))

        # Accept pre-computed sweep_values array OR from/to/step
        if 'sweep_values' in body:
            sweep_values = [float(v) for v in body['sweep_values']]
        else:
            sweep_from = float(body['from'])
            sweep_to = float(body['to'])
            sweep_step = float(body['step'])
            sweep_values = []
            val = sweep_from
            while val <= sweep_to + sweep_step * 0.01:
                sweep_values.append(round(val, 10))
                val += sweep_step

        # MC config: accept mc_enabled+mc_seeds (frontend) or mc (legacy)
        mc_enabled = body.get('mc_enabled', False)
        if mc_enabled:
            mc = int(body.get('mc_seeds', 20))
        else:
            mc = int(body.get('mc', 1))

        # Extract config from nested 'config' key or top-level body
        fixed = body.get('config', body)

        cfg_base = {}
        config_keys = [
            'N', 'T', 'rho', 'eta_bailout', 'alpha_collateral',
            'gamma_capital', 'fiscal_regime', 'fund_levy_rate',
            'fund_initial_balance', 'mu', 'omega', 'beta', 'alfa',
            'phi', 'chi',
        ]
        for k in config_keys:
            if k in fixed and k != sweep_key:
                cfg_base[k] = fixed[k]

        base_seed = int(fixed.get('seed', 26462))
        algorithm = fixed.get('algorithm', 'Boltzmann')
        lc_p = float(fixed.get('lc_p', 0.5))
        lc_m = int(fixed.get('lc_m', 1))

        # Build seeds list
        seeds = [base_seed + s for s in range(mc)]

        def generate():
            # Build job list
            jobs = []
            for val in sweep_values:
                for seed in seeds:
                    jobs.append((cfg_base, sweep_key, val, seed,
                                 algorithm, lc_p, lc_m, metrics))

            total = len(jobs)
            actual_workers = min(workers, total)
            done_count = 0

            all_results = []
            with ProcessPoolExecutor(max_workers=actual_workers) as pool:
                futures = {pool.submit(_run_single_sweep, job): job for job in jobs}
                for future in as_completed(futures):
                    val, seed, row = future.result()
                    all_results.append((val, seed, row))
                    done_count += 1
                    status_msg = f'{sweep_key}={val}'
                    if mc > 1:
                        status_msg += f' seed={seed}'
                    print(f"  [{done_count}/{total}] {status_msg} done")
                    yield json.dumps({
                        'type': 'progress',
                        'done': done_count,
                        'total': total,
                        'status': status_msg
                    }) + '\n'

            if mc <= 1:
                # Single seed per sweep value
                result_map = {val: row for val, seed, row in all_results}
                run_results = [result_map.get(val, {}) for val in sweep_values]

                yield json.dumps({
                    'type': 'result',
                    'mc': False,
                    'sweep_values': sweep_values,
                    'values': run_results
                }) + '\n'
            else:
                # MC averaging: group by sweep value
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
                    'mc_seeds': mc,
                    'sweep_values': sweep_values,
                    'values': agg_results
                }) + '\n'

        return app.response_class(generate(), mimetype='text/plain')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Run same config under multiple fiscal regimes for side-by-side comparison."""
    try:
        body = request.get_json(force=True)

        regimes = body.get('regimes', ['none', 'socialized_tax', 'resolution_fund'])
        metrics = body.get('metrics', [
            'bankruptcy', 'bankruptcies_shock', 'bankruptcies_rationing',
            'bankruptcies_repay', 'bankruptcies_contagion', 'bankruptcies_fiscal',
            'fire_sale_survivors', 'tax_induced_failures',
            'bailout_bill', 'bailout_count',
        ])

        # Extract config from nested 'config' key or top-level body
        fixed = body.get('config', body)

        seed = int(fixed.get('seed', 26462))
        algorithm = fixed.get('algorithm', 'Boltzmann')
        lc_p = float(fixed.get('lc_p', 0.5))
        lc_m = int(fixed.get('lc_m', 1))

        # Build base config (without fiscal_regime)
        config_base = _build_config(fixed)
        config_base.pop('fiscal_regime', None)

        results = {}
        for regime in regimes:
            cfg = dict(config_base)
            cfg['fiscal_regime'] = regime

            model = _create_and_run_model(cfg, seed, algorithm, lc_p, lc_m)
            T = model.t

            regime_metrics = {}
            for m in metrics:
                regime_metrics[m] = extract_metric(model.statistics, m, T)

            results[regime] = regime_metrics

        return jsonify({'regimes': results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save', methods=['POST'])
def api_save():
    """Save sweep results as CSV + PNG to the Simulations folder."""
    try:
        body = request.get_json()
        name = body['name']
        png_b64 = body.get('png', '')
        csv_text = body['csv']

        os.makedirs(SAVE_DIR, exist_ok=True)

        csv_path = os.path.join(SAVE_DIR, name + '.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_text)

        png_path = None
        if png_b64:
            png_path = os.path.join(SAVE_DIR, name + '.png')
            with open(png_path, 'wb') as f:
                f.write(base64.b64decode(png_b64))

        return jsonify({'ok': True, 'csv': csv_path, 'png': png_path})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Zombie Lending Channel Dashboard at http://127.0.0.1:5003")
    app.run(host='127.0.0.1', port=5003, debug=False)
