# Too-Big-To-Fail extension of econo-ml

A TBTF (Too-Big-To-Fail) extension of [`hcastillo/econo-ml`](https://github.com/hcastillo/econo-ml), the agent-based interbank network model of Lenzu & Tedeschi (2012) and Berardi & Tedeschi (2017). This fork adds size-dependent bailout expectations to the bilateral loan cap, a two-state default resolution with a random bailout draw, three configurable fiscal regimes, and statistics for the resulting fire-sale-survivor channel.

# Auxiliary files

- `requirements.txt`: list of the necessary python packages.

# Interbank model

- `interbank.py`: use to execute standalone the Interbank simulation. The same CLI as upstream is preserved; the TBTF parameters are added on top.

  - Command-line usage:

    ``` bash
    interbank.py --log DEBUG --n 150 --t 2000
    interbank.py --save results.gdt eta=0.10 gamma_capital=0.10
    interbank.py --lc ShockedMarket3 --p 0.3
    interbank.py --fast
    ```

  - As a package:

    ``` python
    import interbank
    model = interbank.Model()
    model.config.configure(eta_bailout=0.10, fiscal_regime="socialized_tax")
    model.forward()
    ```

- `interbank_lenderchange.py`: network-formation algorithms (Boltzmann, Preferential, ShockedMarket, RestrictedMarket, SmallWorld). Unchanged interface.
- `interbank_agent.py`: RL agent interface (unchanged).

# What the TBTF extension adds

Six new parameters and one rewritten pricing block. The key distortion is the TBTF term `1 / (1 - b_j * eta)` in the bilateral loan cap, which inflates exposure to large borrowers when bailout coverage `eta` is non-zero.

| Symbol | Code name | Default | Role |
|--------|-----------|---------|------|
| gamma | `gamma_capital` | 0.10 | IRB capital adequacy fraction (eq. 6) |
| eta | `eta_bailout` | 0.85 | Bailout recovery fraction — the policy lever |
| alpha | `alpha_collateral` | 0.05 | Collateral recovery for pricing (eqs. 4, 6, 8). Kept separate from `rho` |
| — | `fiscal_regime` | `"socialized_tax"` | One of `"none"`, `"socialized_tax"`, `"resolution_fund"` |
| tau | `fund_levy_rate` | 1e-5 | Periodic levy on bank assets (`resolution_fund` only) |
| — | `fund_initial_balance` | 0.0 | Starting fund balance |

Core equations (full derivation in `doc/algorithm.tex`):

```
p_j = 1 - E_j / E_max                                           # default prob (eq. 2)
b_j = A_{j, t-1} / A_{max, t-1}                                 # bailout prob (eq. 3)
L_ij = min( (gamma * E_i + p_j * (1 - b_j) * alpha * A_j)
            / (p_j * (1 - b_j * eta)),  C_i )                   # bilateral cap (eq. 6)
E(L|d) = (1 - b_j) * (L_ij - alpha * A_j) + b_j * (1 - eta) * L_ij    # two-state loss (eq. 4)
r_ij = (p_j * E(L|d) + screening_cost) / ((1 - p_j) * L_ij)     # rate (eq. 8)
```

Resolution: each failure draws a Bernoulli bailout with probability `b_j`. In the no-bailout branch the lender absorbs `L_ij - alpha * A_j` and may cascade. In the bailout branch the state covers `eta * L_ij`; the bill is settled at end-of-period under the active fiscal regime.

# Fiscal regimes

The `fiscal_regime` selector picks how the bailout bill is paid:

| `fiscal_regime` | Mechanism |
|-----------------|-----------|
| `"none"` | Bailouts are free. Isolates pure moral hazard. |
| `"socialized_tax"` | End-of-period tax on surviving banks, proportional to assets: `S_k = bill * A_k / sum_m A_m`. |
| `"resolution_fund"` | Pre-funded periodic levy `tau * A_k` builds a war chest. Partial bailout if depleted. |

# Basic usage of the model

```bash
# Single TBTF run
python run_tbtf.py

# 5-seed Monte Carlo
python run_mc.py

# eta sweep
python sweep_eta.py
```

The simulation output directory is configurable via the `TBTF_SIM_DIR` environment variable; defaults to a sibling `Simulations/` folder when unset.

# Interactive GUIs

Three Flask backends with HTML frontends in `templates/`:

```bash
python gui_zombie.py     # fire-sale-survivor dashboard         -> http://127.0.0.1:5003
python gui_sweep.py      # parameter sweep + Monte Carlo        -> http://127.0.0.1:5002
python gui_tbtf.py       # single-run TBTF time series          -> http://127.0.0.1:5001
```

The `gui_zombie.py` name is preserved for backward compatibility; the underlying mechanism is the fire-sale-survivor channel (depleted equity after fire sales).

# RL with Stable Baselines3

Upstream's PPO / TD3 runners and the saved policies are preserved:

```bash
python run_ppo.py
python run_td3.py
python run_mc.py
```

Trained policies live under `models/`. See upstream `doc/README.tex` for the full RL workflow.

# Project structure

```
interbank.py                # main model (TBTF extension)
interbank_lenderchange.py   # network formation
interbank_agent.py          # RL agent interface
exp_runner*.py              # upstream executor base classes
__init__.py                 # package marker

gui_zombie.py               # fire-sale-survivor GUI backend
gui_sweep.py                # parameter sweep GUI backend
gui_tbtf.py                 # single-run TBTF GUI backend
templates/                  # HTML frontends

run_tbtf.py, run_mc.py      # canonical runners
run_ppo.py, run_td3.py      # RL runners
sweep_eta.py                # eta sweep runner

scripts/canonical/          # render_*.py, analyze_*.py, sweep_w58_*.py, ccf_w58_*.py
scripts/exploration/        # smoke_*, dump_*, plot_* compares, ablation sweeps
scripts/utils/              # one-off helpers, dashboard extractors
tests/                      # pytest suite (upstream + TBTF boundary cases)

experiments/                # upstream experiment harness (unchanged)
utils/                      # upstream plotting helpers (unchanged)
doc/                        # algorithm.tex, README.tex + tikz figures
models/                     # pre-trained PPO / TD3 policies
```

# Statistics

In addition to the upstream output (`eta`, `liquidity`, `policy`, `interest_rate`, `bankruptcies`, ...), the TBTF extension records:

- `bankruptcies_shock`, `bankruptcies_rationing`, `bankruptcies_repay`, `bankruptcies_contagion`, `bankruptcies_tax`, `bankruptcies_levy`: per-cause bankruptcy decomposition.
- `bailouts`, `bailout_amount`: count and euro-value of bailouts paid in the period.
- `tax_bill`, `levy_collected`, `fund_balance`: fiscal-regime book-keeping.
- `fire_sale_survivors`: count of banks that survived a fire sale with depleted equity (channel name retained from the early `zombie` coinage for backward compatibility).
- `best_lender_clients`: number of borrowers attached to the largest lender each period (hub-load proxy).

# About this work

This repository is the simulation code underpinning a 2025/26 undergraduate thesis (TFG) at Universitat Jaume I on partial bailouts and interior policy optima in interbank networks. The thesis text itself, its LaTeX sources, and the rendered figures are kept in a separate document repository and are not tracked here.

# Upstream and references

- Upstream codebase: [`hcastillo/econo-ml`](https://github.com/hcastillo/econo-ml).
- Lenzu, S., Tedeschi, G. (2012). Systemic risk on different interbank network topologies. *Physica A* 391(18).
- Berardi, S., Tedeschi, G. (2017). From banks' strategies to financial (in)stability. *International Review of Economics & Finance* 47.
- Brini, A., Tedeschi, G., Tantari, D. (2023). Reinforcement learning policy recommendation for interbank network stability. *Journal of Financial Stability* 67.
