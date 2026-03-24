#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web GUI v3 for the Interbank ABM Simulator.
Adds age-rate scatter plot diagnostic (Section 5 of thesis spec).
Run:  python gui_v3.py
Then open http://127.0.0.1:5002 in a browser.
"""
import matplotlib
matplotlib.use('Agg')

import io
import base64
import math
import traceback

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, render_template, request, jsonify

import interbank
import interbank_lenderchange as lc

app = Flask(__name__)

VALID_ALGORITHMS = [
    'Boltzmann', 'InitialStability', 'Preferential',
    'RestrictedMarket', 'ShockedMarket', 'ShockedMarket2',
    'ShockedMarket3', 'SmallWorld',
]

# All statistics to extract
STAT_NAMES = [
    # Balance sheet
    'equity', 'deposits', 'reserves', 'liquidity', 'loans',
    # Market activity
    'interest_rate', 'profits', 'B', 'rationing',
    # Risk & stability
    'prob_bankruptcy', 'bankruptcy', 'leverage', 'maxE',
    # Network & fitness
    'fitness', 'active_lenders', 'active_borrowers', 'num_banks',
    # Extra
    'num_loans', 'incrementD',
]


def safe_list(arr, length):
    """Convert a numpy array slice to a JSON-safe Python list (NaN/Inf → None)."""
    values = arr[:length].tolist()
    return [
        None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
        for v in values
    ]


def build_network_image(model):
    """Render the final bank→lender network as a base64-encoded PNG."""
    G = nx.DiGraph()
    for bank in model.banks:
        G.add_node(bank.id)
        if bank.lender is not None and bank.lender < len(model.banks):
            G.add_edge(bank.id, bank.lender)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, 'No banks remaining', ha='center', va='center',
                fontsize=14, color='#666')
    else:
        pos = nx.spring_layout(G, seed=42)
        colors = []
        for node in G.nodes():
            if model.banks[node].active_borrowers:
                colors.append('#e67e22')
            elif model.banks[node].failed:
                colors.append('#cccccc')
            else:
                colors.append('#3498db')
        nx.draw(G, pos, ax=ax, with_labels=True, arrows=True,
                arrowstyle='->', arrowsize=15,
                node_color=colors, node_size=500,
                font_size=9, font_color='white', font_weight='bold',
                edge_color='#999', width=1.5)
    ax.set_title('Bank Network (borrower \u2192 lender)', fontsize=11, color='#333')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def collect_snapshot(model):
    """Collect per-bank data for the age-rate scatter at the current period."""
    banks_data = []
    N = len(model.banks)

    # System-wide quantities (recomputed fresh)
    T_max = max((b.age for b in model.banks), default=1)
    if T_max == 0:
        T_max = 1

    # r values: bank.r is the avg interest rate charged by bank i
    r_values = [b.r for b in model.banks if hasattr(b, 'r') and b.r > 0]
    r_min = min(r_values) if r_values else 1.0
    r_max = max(r_values) if r_values else 1.0

    # L_max from c_avg_ir (loan amounts L^{i,j})
    L_max = 0
    for bank in model.banks:
        if hasattr(bank, 'c_avg_ir') and isinstance(bank.c_avg_ir, list):
            for val in bank.c_avg_ir:
                if val > L_max:
                    L_max = val
    if L_max == 0:
        L_max = 1.0

    for bank in model.banks:
        if bank.failed:
            continue

        age = getattr(bank, 'age', 0)
        r = getattr(bank, 'r', 0)
        if r <= 0:
            continue

        # Normalized reputation: T_i / T_max
        reputation = age / T_max

        # Normalized price: r_min / r_i (higher = cheaper = better)
        price = r_min / r if r > 0 else 0

        # Average omega (exposure weight) for this bank
        omega_avg = 0.5
        if hasattr(bank, 'c_avg_ir') and isinstance(bank.c_avg_ir, list):
            omega_sum = 0
            omega_count = 0
            for j in range(len(bank.c_avg_ir)):
                if j != bank.id and bank.c_avg_ir[j] > 0:
                    omega_sum += bank.c_avg_ir[j] / L_max
                    omega_count += 1
            if omega_count > 0:
                omega_avg = omega_sum / omega_count

        # Number of active borrowers
        n_borrowers = len(bank.active_borrowers) if isinstance(bank.active_borrowers, dict) else 0

        banks_data.append({
            'id': bank.id,
            'age': age,
            'r': r,
            'reputation': reputation,
            'price': price,
            'omega': omega_avg,
            'n_borrowers': n_borrowers,
        })

    return {
        'period': model.t,
        'banks': banks_data,
        'T_max': T_max,
        'r_min': r_min,
        'r_max': r_max,
        'L_max': L_max,
    }


@app.route('/')
def index():
    return render_template('index_v3.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        params = request.get_json(force=True)

        N = max(2, int(params.get('N', 5)))
        T = max(1, int(params.get('T', 10)))
        seed = int(params.get('seed', 26462))
        algorithm = params.get('algorithm', 'Boltzmann')
        if algorithm not in VALID_ALGORITHMS:
            algorithm = 'Boltzmann'
        lc_p = float(params.get('lc_p', 0.5))
        lc_m = int(params.get('lc_m', 1))

        config_params = {'N': N, 'T': T}
        float_keys = [
            'sigma_min', 'delta_min', 'gamma_screening', 'alpha_recovery',
            'beta', 'mu', 'omega', 'rho',
        ]
        for key in float_keys:
            if key in params:
                config_params[key] = float(params[key])

        # --- Run simulation with snapshot collection ---
        model = interbank.Model()
        model.configure(**config_params)
        model.config.lender_change = lc.determine_algorithm(algorithm, p=lc_p, m=lc_m)
        model.test = True
        model.initialize(seed=seed, generate_plots=False)

        # Determine snapshot periods: every T/10 + final
        snapshot_interval = max(1, T // 10)
        snapshot_periods = set(range(0, T, snapshot_interval))
        snapshot_periods.add(T - 1)

        snapshots = []
        for step in range(T):
            model.forward()
            if step in snapshot_periods:
                snapshots.append(collect_snapshot(model))
            # Early stop if too few banks
            if not model.config.allow_replacement_of_bankrupted and len(model.banks) <= 2:
                model.config.T = model.t
                # Collect final snapshot if not already
                if step not in snapshot_periods:
                    snapshots.append(collect_snapshot(model))
                break

        # --- Collect time-series statistics ---
        t = model.t
        result = {'time': list(range(t))}

        for name in STAT_NAMES:
            arr = getattr(model.statistics, name, None)
            if arr is not None:
                result[name] = safe_list(arr, t)

        # Cumulative bankruptcy
        bankruptcy_arr = model.statistics.bankruptcy[:t]
        result['bankruptcy_cumulative'] = np.cumsum(bankruptcy_arr).tolist()

        # Boltzmann switching probability
        result['has_boltzmann'] = model.statistics.P is not None
        if model.statistics.P is not None:
            result['P'] = safe_list(model.statistics.P, t)
            result['P_max'] = safe_list(model.statistics.P_max, t)
            result['P_min'] = safe_list(model.statistics.P_min, t)

        # --- Network graph ---
        result['network'] = build_network_image(model)

        # --- Scatter snapshots ---
        result['snapshots'] = snapshots

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Interbank ABM GUI v3 at http://127.0.0.1:5002")
    app.run(debug=True, port=5002)
