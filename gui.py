#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web GUI for the Interbank ABM Simulator.
Run:  python gui.py
Then open http://127.0.0.1:5000 in a browser.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be before any other matplotlib import

import io
import base64
import math
import traceback

import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, render_template, request, jsonify

import interbank
import interbank_lenderchange as lc

app = Flask(__name__)

# Valid algorithm names for the lender-change mechanism dropdown
VALID_ALGORITHMS = [
    'Boltzmann', 'InitialStability', 'Preferential',
    'RestrictedMarket', 'ShockedMarket', 'ShockedMarket2',
    'ShockedMarket3', 'SmallWorld',
]

# Statistics to extract and send to the frontend
STAT_NAMES = [
    'interest_rate', 'equity', 'bankruptcy', 'fitness',
    'liquidity', 'loans', 'deposits', 'leverage',
    'rationing', 'profits', 'prob_bankruptcy',
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
        # Color nodes: banks with borrowers (lenders) in orange, others in blue
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
    ax.set_title('Bank Network (borrower → lender)', fontsize=11, color='#333')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        params = request.get_json(force=True)

        # --- Extract and validate parameters ---
        N = max(2, int(params.get('N', 5)))
        T = max(1, int(params.get('T', 10)))
        seed = int(params.get('seed', 26462))
        algorithm = params.get('algorithm', 'Boltzmann')
        if algorithm not in VALID_ALGORITHMS:
            algorithm = 'Boltzmann'

        # Config-level float parameters (thesis spec)
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
        model.config.lender_change = lc.determine_algorithm(algorithm)
        model.test = True  # suppress finish() side-effects if called accidentally
        model.initialize(seed=seed, generate_plots=False)
        model.simulate_full()

        # --- Collect statistics ---
        result = {'time': list(range(model.t))}
        for name in STAT_NAMES:
            arr = getattr(model.statistics, name, None)
            if arr is not None:
                result[name] = safe_list(arr, model.t)

        # --- Network graph ---
        result['network'] = build_network_image(model)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Interbank ABM GUI at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
