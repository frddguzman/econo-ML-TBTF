# CLAUDE.md — Interbank ABM: Endogenous Reputation & Exposure-Driven Network Formation

## Project overview
Undergraduate economics thesis at Universitat Jaume I (IEI lab). Adapting the econo-ml
interbank ABM (`interbank.py`, `interbank_lenderchange.py`) to implement an endogenous
reputation/exposure-driven network formation model. The formal spec is in `thesis_spec.tex`.

The core contribution is replacing the original lender selection and interest rate logic
with a microfounded fitness function φ_i that trades off institutional reputation (age)
against price (interest rate), with an endogenous exposure-driven weight ω_{i,j}.

---

## Relevant files
- `interbank.py` — balance sheet mechanics, default/bankruptcy, interest rate. **Primary file.**
- `interbank_lenderchange.py` — Boltzmann lender switching + any mechanics coupled to
  network formation. **Relevant: Boltzmann switching logic and fitness evaluation.**

## Files to ignore
All ML/neural network components in the repo. Do not touch, import from, or suggest
changes to any file outside the two listed above.

---

## Model spec (summary — always defer to thesis_spec.tex for exact formulas)

### Balance sheet (eq. 1)
    A^j = D^j + E^j
Assets decompose into liquid reserves C^j and illiquid assets A^j_up.

### Default probability (eq. 2)
    p_j = 1 - E^j / E_max
- p_j → 1 as equity → 0 (near-insolvent)
- p_j → 0 as equity → E_max (most capitalized)
Maps to: `interbank.py`, wherever bank fragility / default probability is computed.

### Screening cost function (eqs. 3–7)
    σ(T_i) = σ_min + γ * ln(1 + T_i)        # lender cost: increases with age
    δ(T_j) = δ_min + γ * ln(1 + T_j)        # borrower transparency: reduces cost
    K_{i,j} = σ(T_i)*A^i - δ(T_j)*A^j      # total screening cost

Key invariants (must hold in code):
    ∂K/∂T_i > 0   (older lender → higher cost)
    ∂K/∂T_j < 0   (older borrower → lower cost)

Parameters: σ_min, δ_min (exogenous lower bounds), γ (universal maturity rate).
Maps to: new method in `interbank.py`, called during loan evaluation.

### Expected profit (eq. 8)
    E[Π^{i,j}] = (1-p_j)*r^{i,j}*L^{i,j} + p_j*(α*A^j - L^{i,j}) - K_{i,j}

where α ∈ [0,1] is the collateral recovery rate.

### Interest rate — zero profit condition (eq. 9)
    r^{i,j} = [ K_{i,j} - p_j*(α*A^j - L^{i,j}) ] / [ (1-p_j)*L^{i,j} ]

Derived by setting E[Π^{i,j}] = 0 and solving for r^{i,j}.
Key invariant: ∂r/∂T_i > 0 (older lender charges more).
Maps to: interest rate method in `interbank.py`.

### Haircut and loan size (eq. 10)
    λ^j = A^j_LP / E^j              # leverage ratio
    h^j = λ^j / λ_max               # normalized haircut
    L^{i,j} = (1 - h^j) * A^j > 0  # loan constrained by collateral

Maps to: loan sizing logic in `interbank.py`.

### Fitness function (eq. 11)
    φ_i = ω_{i,j} * (T_i / T_max) + (1 - ω_{i,j}) * (r_min / r^{i,j})
              ^-- Reputation              ^-- Price (inverse)

Structural property: T_i↑ pushes reputation↑ but price↓ — the two vectors are
antithetical, so the network does NOT trivially converge to oldest banks.
Maps to: fitness evaluation in `interbank_lenderchange.py`.

### Exposure weight (eq. 12)
    ω_{i,j} = L^{i,j} / L_max

- Large loan → ω → 1 → reputation dominates (borrower picks safe/old lender)
- Small loan → ω → 0 → price dominates (borrower shops for cheapest rate)
Maps to: computed before fitness evaluation, in `interbank_lenderchange.py`.

### Boltzmann switching (eq. 13)
    P(i → k) = 1 / (1 + exp(-β * (φ_k - φ_i)))

β > 0 is intensity of choice. High β → near-deterministic; moderate β → bounded rationality.
Maps to: Boltzmann switching method in `interbank_lenderchange.py`. Preserve structure,
only replace the fitness input with φ_i as defined above.

---

## Parameters to track
| Symbol  | Role                                       | Location         |
|---------|--------------------------------------------|------------------|
| γ       | Universal institutional maturity rate      | interbank.py     |
| σ_min   | Baseline lender cost                       | interbank.py     |
| δ_min   | Baseline borrower transparency             | interbank.py     |
| α       | Collateral recovery rate                   | interbank.py     |
| β       | Boltzmann intensity of choice              | lenderchange.py  |
| L_max   | System max loan (updated each period)      | both files       |
| T_max   | System max age (updated each period)       | lenderchange.py  |
| r_min   | System min interest rate (updated each t)  | lenderchange.py  |
| λ_max   | System max leverage (updated each period)  | interbank.py     |

All system-wide max/min quantities (L_max, T_max, r_min, λ_max, E_max) must be
recomputed at the start of each period t, not cached from t-1.

---

## Priorities (in order)
1. **Faithfulness to thesis_spec.tex** — if spec says X, code implements X. Flag any
   ambiguity or inconsistency rather than resolving silently.
2. **Minimal changes vs original** — preserve existing class structure, method signatures,
   and variable naming conventions wherever compatible with the spec. Do not refactor.
3. Inline comments mapping every modified method to its equation number.

## What NOT to do
- Do not rename or restructure existing methods unless the spec structurally requires it.
- Do not silently "fix" anything that looks like a modeling choice — ask first.
- Do not touch ML components.
- Do not cache system-wide statistics across periods.

---

## Workflow
Local development in VSCode
he repo is the single source of truth.