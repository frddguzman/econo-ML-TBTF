#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web GUI v2 for the Interbank ABM Simulator.
Improved charts: grouped panels, correct chart types, more statistics.
Run:  python gui_v2.py
Then open http://127.0.0.1:5001 in a browser.
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


@app.route('/')
def index():
    return render_template('index_v2.html')


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

        # --- Run simulation ---
        model = interbank.Model()
        model.configure(**config_params)
        model.config.lender_change = lc.determine_algorithm(algorithm, p=lc_p, m=lc_m)
        model.test = True
        model.initialize(seed=seed, generate_plots=False)
        model.simulate_full()

        # --- Collect statistics ---
        t = model.t
        result = {'time': list(range(t))}

        for name in STAT_NAMES:
            arr = getattr(model.statistics, name, None)
            if arr is not None:
                result[name] = safe_list(arr, t)

        # Cumulative bankruptcy
        bankruptcy_arr = model.statistics.bankruptcy[:t]
        result['bankruptcy_cumulative'] = np.cumsum(bankruptcy_arr).tolist()

        # Boltzmann switching probability (only exists for Boltzmann-based mechanisms)
        result['has_boltzmann'] = model.statistics.P is not None
        if model.statistics.P is not None:
            result['P'] = safe_list(model.statistics.P, t)
            result['P_max'] = safe_list(model.statistics.P_max, t)
            result['P_min'] = safe_list(model.statistics.P_min, t)

        # --- Network graph ---
        result['network'] = build_network_image(model)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Interbank ABM GUI v2 at http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
