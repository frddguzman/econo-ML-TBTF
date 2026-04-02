"""
Full run with exact GUI parameters — produces .gdt, CSV, and plots.
Run from econo-ml-main/:  python run_tbtf.py
"""
import interbank
import interbank_lenderchange as lc

model = interbank.Model()

model.configure(
    N=50, T=1000,
    # TBTF / Bailout
    gamma_capital=0.10,
    eta_bailout=0.85,
    alpha_collateral=0.20,
    # Screening costs
    phi=0.025,
    chi=0.015,
    # Recovery & Switching
    xi=0.3,
    rho=0.3,
    beta=5,
    alfa=0.1,
    # Shocks
    mu=0.7,
    omega=0.5,
)
model.config.lender_change = lc.determine_algorithm('Boltzmann', p=0.5, m=1)

# Output: gdt + plots as pdf (change 'pdf' to 'png', 'svg', or 'none')
model.statistics.define_output_format('gdt')
model.statistics.define_plot_format('pdf')

model.initialize(seed=26462, export_datafile='results', output_directory='output')
model.simulate_full(interactive=True)
model.finish()
