# Changelog: TBTF Extension

All changes relative to the upstream [`hcastillo/econo-ml`](https://github.com/hcastillo/econo-ml) baseline, which implements:

- Lenzu & Tedeschi (2012) -- *Systemic risk on different interbank network topologies* (Physica A)
- Berardi & Tedeschi (2017) -- *From banks' strategies to financial (in)stability* (IREF)

The TBTF extension implements the mathematical model specified in [`alternativa.tex`](alternativa.tex).

---

## 1. New Configuration Parameters

Six new parameters added to the `Config` class to support TBTF mechanics:

| Parameter | Code name | Default | Role |
|-----------|-----------|---------|------|
| gamma | `gamma_capital` | 0.08 | IRB capital adequacy fraction -- controls the bilateral exposure cap (eq. 6) |
| eta | `eta_bailout` | 0.85 | Bailout recovery fraction -- the core TBTF policy instrument |
| alpha | `alpha_collateral` | 0.05 | Collateral recovery rate for **pricing only** (eqs. 4, 6, 8). Kept separate from rho (actual resolution) |
| Fiscal regime | `fiscal_regime` | `"socialized_tax"` | Selector: `"none"`, `"socialized_tax"`, or `"resolution_fund"` |
| Fund levy rate | `fund_levy_rate` | 0.005 | Periodic levy on bank assets (resolution fund regime) |
| Fund initial balance | `fund_initial_balance` | 0.0 | Starting resolution fund balance |

---

## 2. Core Change: `do_interest_rate()` -- Complete Rewrite

**Location:** `interbank.py`, Model class, ~line 1939

The baseline BT 2017 pricing formula:
```
r_ij = [(chi*A_i - phi*A_j - (1-p_j)*(xi*A_j - c)) * (1+psi)] / (p_j * c)
```

was replaced entirely with the TBTF equations (eqs. 2--9 of `alternativa.tex`):

**Eq. 2 -- Default probability** (unchanged from baseline concept):
```
p_j = 1 - E_j / E_max
```

**Eq. 3 -- Bailout probability** (new):
```python
b_j = bank_j.A_lagged / self.max_A_lagged
```
Uses **lagged** assets (one period behind) to prevent simultaneity.

**Eq. 6 -- Bilateral loan cap with TBTF distortion** (new):
```python
L_ij = min(
    (gamma * E_i + p_j * (1 - b_j) * alpha * A_j)
    / (p_j * (1 - b_j * eta)),
    bank_i.C
)
```
**This is the key distortion equation.** The denominator `(1 - b_j * eta)` shrinks when `b_j` is high (large bank) and `eta > 0` (bailout expected), inflating the loan cap. This causes disproportionate credit to flow toward TBTF borrowers.

**Eq. 4 -- Two-state expected loss** (new):
```python
EL_given_d = (1 - b_j) * (L_ij - alpha * A_j) + b_j * (1 - eta) * L_ij
```
Two states: no-bailout (recover `alpha * A_j` as collateral) and bailout (state absorbs `eta` fraction).

**Eq. 8 -- Interest rate** (rewritten):
```python
r_ij = (p_j * EL_given_d + screening_cost) / ((1 - p_j) * L_ij)
```
Zero-profit condition. BT 2017 screening costs (`chi * A_i - phi * A_j`) are preserved.

**Eq. 9 -- Fitness** (simplified):
```python
bank_i.mu = bank_i.E / self.maxE
```

**Boundary cases** added for numerical stability:
- `p_j <= 0` (perfectly capitalized): `L_ij = C_i`, `r_ij = screening_cost / C_i`
- `p_j >= 1` (insolvent): `L_ij = 0`, `r_ij = infinity`
- `(1 - b_j * eta) <= 0` (degenerate): `L_ij = 0`, `r_ij = infinity`

**New per-bank storage:** `bank_i.L_ij_max[j]` -- bilateral cap from eq. 6 (enforced in `do_loans()`).

---

## 3. Core Change: `do_bankruptcy()` -- Two-State TBTF Resolution

**Location:** `interbank.py`, Bank class, ~line 2131

The baseline had a **single resolution path**: liquidate at rho, pay depositors first, residual to lender.

The TBTF extension adds a **two-state resolution** with a random bailout draw:

### Bankruptcy phase decomposition (new)
Each failure is categorized by cause:
- `shock` -- failed during deposit shock
- `rationing` -- failed from insufficient credit
- `repay` -- failed during loan repayment
- `contagion` -- failed from lender's bad-debt cascade
- `tax` / `levy` -- failed from fiscal burden

### Bailout probability check
```python
b_j = self.A_lagged / self.model.max_A_lagged
if eta > 0 and random.random() < b_j:
    # BAILOUT STATE
else:
    # NO-BAILOUT STATE
```

### Bailout state (new)
When bailed out, the handling depends on the fiscal regime:

- **`"resolution_fund"`**: draws from pre-funded balance; partial bailout if depleted
- **`"socialized_tax"`**: accumulates cost in `period_bailout_bill` for end-of-period taxation
- **`"none"`**: free bailout, no fiscal cost

```python
bailout_amount = eta * self.l
bad_debt = (1 - eta) * self.l  # residual loss to lender
```

### No-bailout state (restored from baseline)
Uses the professor's depositor-priority waterfall:
```python
liquidation_proceeds = rho * self.L
after_depositors = max(liquidation_proceeds - self.D, 0)
recovered_for_lender = min(after_depositors, self.l)
bad_debt = self.l - recovered_for_lender
```
At baseline parameters (`rho=0.4`, `L=120`, `D=135`): proceeds = 48, depositors claim 135, lender recovers 0. Bad debt equals the full loan -- making TBTF failure catastrophic.

### One-hop contagion (tracked separately)
```python
lender.E -= bad_debt
if lender.E < 0:
    lender.do_bankruptcy('contagion')
```

---

## 4. Modified: `do_loans()` -- Bilateral Cap Enforcement

**Location:** `interbank.py`, Model class, ~line 1481

The baseline had no per-borrower lending limit:
```python
effective_supply = min(lender.s, borrower.d)
```

The TBTF extension enforces the IRB bilateral cap from eq. 6:
```python
L_ij_cap = lender.L_ij_max[borrower.id]
effective_supply = min(lender.s, L_ij_cap)
```

This is what makes zombie banks (low `E_i` -> low `gamma * E_i`) toxic to the network: they can only lend tiny amounts, rationing their borrowers and triggering fire-sale cascades.

---

## 5. Modified: `do_fire_sales()` -- Zombie Tracking

**Location:** `interbank.py`, Bank class, ~line 2235

Added at the end of fire-sale resolution:
```python
if not self.failed:
    self.model.statistics.fire_sale_survivors[self.model.t] += 1
```

A bank that survives a fire sale but with depleted equity and assets is a "zombie." Zombies degrade network lending capacity through the bilateral cap mechanism (section 4).

---

## 6. Modified: `forward()` -- Fiscal Regime Step + Lagged Assets

**Location:** `interbank.py`, Model class, ~line 1272

Two new blocks inserted into the simulation step:

### Fiscal regime conditional (after `do_repayments()`, before `replace_bankrupted_banks()`):
```python
if self.config.fiscal_regime == "socialized_tax":
    self.apply_bailout_tax()
elif self.config.fiscal_regime == "resolution_fund":
    self.collect_fund_levy()
```

### Lagged asset update (end of period):
```python
for bank in self.banks:
    if not bank.failed:
        bank.A_lagged = bank.A
```
Enables eq. 3's one-period lag -- a bank's current borrowing can't inflate its own bailout probability.

---

## 7. New Method: `apply_bailout_tax()`

**Location:** `interbank.py`, Model class, ~line 1645

Implements the **socialized ex-post tax** regime. After all defaults in a period:

1. Snapshot accumulated `period_bailout_bill` and zero it
2. Distribute proportionally to surviving assets: `tax = bill * (A_k / total_A)`
3. Banks pay from cash first; shortfalls trigger fire sales
4. If fire sales push equity below threshold: bank fails (`bankruptcies_fiscal`)

Tax-triggered cascades accumulate fresh bailout costs carried to the next period (no double-taxation within a round).

---

## 8. New Method: `collect_fund_levy()`

**Location:** `interbank.py`, Model class, ~line 1713

Implements the **pre-funded resolution fund** regime:

1. Each surviving bank pays: `levy = fund_levy_rate * A_k`
2. Cash-first payment; shortfalls fire-sold
3. Collected amounts credited to `resolution_fund_balance`
4. Fund balance available for bailouts **next period** (one-period lag)
5. If levy pushes equity below threshold: bank fails (`bankruptcies_fiscal`)

---

## 9. New Statistics Arrays

Thirteen new per-period tracking arrays in the `Statistics` class:

### Bankruptcy decomposition:
| Array | Purpose |
|-------|---------|
| `bankruptcies_shock` | Failures from deposit shocks |
| `bankruptcies_rationing` | Failures from credit rationing |
| `bankruptcies_repay` | Failures during loan repayment |
| `bankruptcies_contagion` | Failures from bad-debt contagion |
| `bankruptcies_fiscal` | Failures from tax or levy payment |

### Bailout activity:
| Array | Purpose |
|-------|---------|
| `bailout_bill` | Total bailout cost per period |
| `bailout_count` | Number of bailouts triggered per period |
| `bailout_tax_total` | Tax collected from surviving banks |
| `tax_induced_failures` | Banks killed by bailout tax |

### Fire sales and resolution fund:
| Array | Purpose |
|-------|---------|
| `fire_sale_survivors` | Zombie count (fire-sale survivors with depleted balance sheets) |
| `resolution_fund_balance` | Fund balance per period |
| `fund_depleted_events` | Times the fund ran dry |
| `total_levy_collected` | Levy revenue per period |

---

## 10. Bug Fix: The alpha / xi / rho Confusion

### The problem
The baseline uses two separate parameters for two separate roles:

| Parameter | Role | Where used |
|-----------|------|------------|
| xi (0.3) | Collateral recovery for **pricing** | `do_interest_rate()` |
| rho | Fire-sale price for **actual resolution** | `do_fire_sales()` and `do_bankruptcy()` |

The TBTF extension introduced `alpha_collateral` which correctly replaced xi in the pricing equations (eqs. 4, 6, 8) but **incorrectly also replaced rho in `do_bankruptcy()`**. The no-bailout branch used `alpha * A_lagged` for resolution recovery, when actual resolution should use the depositor-priority waterfall with rho.

### The fix
In `do_bankruptcy()`, the no-bailout branch now uses:
```python
liquidation_proceeds = self.model.config.rho * self.L
after_depositors = max(liquidation_proceeds - self.D, 0)
recovered_for_lender = min(after_depositors, self.l)
bad_debt = self.l - recovered_for_lender
```
This restores the catastrophic nature of TBTF failure for lenders.

---

## 11. New Bank Instance Variables

| Variable | Purpose |
|----------|---------|
| `A_lagged` | Previous period's total assets -- used in eq. 3 for bailout probability |
| `L_ij_max` | List of bilateral loan caps per borrower -- computed in `do_interest_rate()` via eq. 6 |

---

## 12. New GUIs

Three Flask-based web interfaces for interactive analysis:

| File | Port | Purpose |
|------|------|---------|
| `gui_zombie.py` | 5003 | Zombie channel dashboard -- single runs, parameter sweeps, regime comparisons |
| `gui_sweep.py` | 5002 | General parameter sweep GUI with Monte Carlo averaging |
| `gui_tbtf.py` | 5001 | Single-simulation TBTF GUI with time-series visualization |

HTML templates in `templates/`: `index_zombie.html`, `index_sweep.html`, `index_tbtf.html`.

---

## 13. New Files

| File | Purpose |
|------|---------|
| `alternativa.tex` | Full mathematical specification of the TBTF extension (eqs. 1--10 + comparative statics) |
| `sweep_eta.py` | CLI script for eta parameter sweeps |
| `run_tbtf.py` | Single TBTF simulation runner |
| `doc/figures/*.tex` | TikZ source for thesis diagrams |
| `doc/figures/*.png` | Rendered diagram PNGs |
