#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web GUI for the Interbank ABM Simulator — TBTF extension.
Run:  python gui_tbtf.py
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

# All statistics to extract (including TBTF)
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
    # TBTF
    'bailout_bill', 'bailout_count', 'bailout_tax_total', 'tax_induced_failures',
]


def safe_list(arr, length):
    """Convert a numpy array slice to a JSON-safe Python list (NaN/Inf -> None)."""
    values = arr[:length].tolist()
    return [
        None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
        for v in values
    ]


def build_network_image(model):
    """Render the final bank->lender network as a base64-encoded PNG."""
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
            b = model.banks[node]
            if b.failed:
                colors.append('#cccccc')
            elif hasattr(b, 'active_borrowers') and b.active_borrowers:
                colors.append('#e67e22')
            else:
                colors.append('#3498db')

        # Size nodes by total assets (bigger = more TBTF)
        max_a = max((model.banks[n].A for n in G.nodes() if not model.banks[n].failed), default=1)
        sizes = []
        for node in G.nodes():
            b = model.banks[node]
            if b.failed or max_a <= 0:
                sizes.append(300)
            else:
                sizes.append(300 + 700 * (b.A / max_a))

        nx.draw(G, pos, ax=ax, with_labels=True, arrows=True,
                arrowstyle='->', arrowsize=15,
                node_color=colors, node_size=sizes,
                font_size=9, font_color='white', font_weight='bold',
                edge_color='#999', width=1.5)
    ax.set_title('Bank Network (borrower -> lender)\nNode size ~ total assets (TBTF)',
                 fontsize=11, color='#333')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def collect_tbtf_snapshot(model):
    """Per-bank snapshot: equity, assets, bailout probability, interest rate."""
    banks_data = []
    max_a_lagged = getattr(model, 'max_A_lagged', 1.0) or 1.0
    max_e = getattr(model, 'maxE', 1.0) or 1.0

    for bank in model.banks:
        if bank.failed:
            continue
        A = getattr(bank, 'A', 0)
        A_lag = getattr(bank, 'A_lagged', A)
        E = getattr(bank, 'E', 0)
        r = getattr(bank, 'r', 0)
        p_j = 1 - (E / max_e) if max_e > 0 else 1.0
        b_j = A_lag / max_a_lagged if max_a_lagged > 0 else 0

        n_borrowers = sum(
            1 for b in model.banks
            if not b.failed and b.id != bank.id and b.get_lender() is not None and b.get_lender().id == bank.id
        )

        banks_data.append({
            'id': bank.id,
            'E': round(E, 4),
            'A': round(A, 4),
            'A_lagged': round(A_lag, 4),
            'r': round(r, 6),
            'p_j': round(p_j, 4),
            'b_j': round(b_j, 4),
            'phi': round(E / max_e, 4) if max_e > 0 else 0,
            'n_borrowers': n_borrowers,
        })

    return {
        'period': model.t,
        'banks': banks_data,
        'max_E': round(max_e, 4),
        'max_A_lagged': round(max_a_lagged, 4),
    }


@app.route('/')
def index():
    return render_template('index_tbtf.html')


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

        # Map GUI parameter names to Config attribute names
        float_keys = {
            # Screening costs
            'phi': 'phi',
            'chi': 'chi',
            # Recovery & switching
            'rho': 'rho',
            'beta': 'beta',
            'alfa': 'alfa',
            # Shocks
            'mu': 'mu',
            'omega': 'omega',
            # TBTF parameters
            'gamma_capital': 'gamma_capital',
            'eta_bailout': 'eta_bailout',
            'alpha_collateral': 'alpha_collateral',
        }
        for gui_key, config_key in float_keys.items():
            if gui_key in params:
                config_params[config_key] = float(params[gui_key])

        # Fiscal regime (string parameter)
        if 'fiscal_regime' in params:
            config_params['fiscal_regime'] = str(params['fiscal_regime'])
        # Resolution fund parameters
        for k in ('fund_levy_rate', 'fund_initial_balance'):
            if k in params:
                config_params[k] = float(params[k])

        # --- Run simulation with snapshot collection ---
        model = interbank.Model()
        model.configure(**config_params)
        model.config.lender_change = lc.determine_algorithm(algorithm, p=lc_p, m=lc_m)
        model.test = True
        model.initialize(seed=seed, generate_plots=False)

        # Snapshot every T/10 + final
        snapshot_interval = max(1, T // 10)
        snapshot_periods = set(range(0, T, snapshot_interval))
        snapshot_periods.add(T - 1)

        snapshots = []
        for step in range(T):
            model.forward()
            if step in snapshot_periods:
                snapshots.append(collect_tbtf_snapshot(model))
            if not model.config.allow_replacement_of_bankrupted and len(model.banks) <= 2:
                model.config.T = model.t
                if step not in snapshot_periods:
                    snapshots.append(collect_tbtf_snapshot(model))
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

        # Cumulative bailout bill
        bill_arr = model.statistics.bailout_bill[:t]
        result['bailout_bill_cumulative'] = np.cumsum(bill_arr).tolist()

        # Boltzmann switching probability
        result['has_boltzmann'] = model.statistics.P is not None
        if model.statistics.P is not None:
            result['P'] = safe_list(model.statistics.P, t)
            result['P_max'] = safe_list(model.statistics.P_max, t)
            result['P_min'] = safe_list(model.statistics.P_min, t)

        # --- Network graph ---
        result['network'] = build_network_image(model)

        # --- TBTF scatter snapshots ---
        result['snapshots'] = snapshots

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Interbank ABM TBTF GUI at http://127.0.0.1:5002")
    app.run(debug=True, port=5002)
