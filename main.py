import numpy as np
import time
from Resources.create_data_pf import return_knowns_guesses
from Resources.initial_conditions_calculation import compute_initial_conditions
from Resources.dynamic_solver import Trapezoidal_dae_multitimestep
from Resources.disturbance_conditions import contingency_compiler
from Resources.neural_net_upload import define_neural_net
from Machine_models.sauer_pi_models import Machine_Models
from Machine_models.machine_types_parameters import return_machine_parameters
from Cases.file_define_bus_system import case_pglib_sys
from Post_processing.plotting_functions import trajectories_overview
from Utils.run_power_flow_last_conditons import convergence_solution, load_power_flow
from Utils.saving_file import save_dynamic_simulation
from Utils.printing_options import print_sim_time_polished

# System Description
study_case = "case30_ieee"
study_case_sys = case_pglib_sys(study_case)
Y_adm = study_case_sys.return_admittance_matrix()
slack_bus, pv_buses, pg_buses, vars_ini_arrays = study_case_sys.define_buses_type()
apply_mach_sauer_pi = True
if apply_mach_sauer_pi:
    vars_ini_arrays = study_case_sys.apply_same_machine_in_sys(vars_ini_arrays)
vm_array_ini, va_array_ini, pg_array_ini, qg_array_ini, pl_array_ini, ql_array_ini = vars_ini_arrays
machine_types = study_case_sys.return_machine_types()

# Steady-State Power Flow Tool
knowns, guesses = return_knowns_guesses((slack_bus, pv_buses, pg_buses), vars_ini_arrays)
final_active_p, final_reactive_p, final_volt_magn, final_volt_angle = load_power_flow(study_case)

vm_array_final = final_volt_magn
va_array_final = final_volt_angle
pg_array_final = final_active_p   + pl_array_ini
qg_array_final = final_reactive_p + ql_array_ini
pl_array_final = pl_array_ini
ql_array_final = ql_array_ini
sl_array_final = pl_array_final + 1j*ql_array_final

simplification_transient_saliency = True
assert len(machine_types) == 1 + pv_buses.shape[0]
m_parameters_class = return_machine_parameters(machine_types)
H, Rs, Xd, Xd_p, Xq, Xq_p, Tdo_p, Tqo_p, dampings = m_parameters_class.return_machine_params(simplification_transient_saliency)
KA_vec, TA_vec, KE_vec, TE_vec, KF_vec, TF_vec = m_parameters_class.return_params_avr()
stator_ready_matrices = m_parameters_class.return_stator_matrices()
nominal_freq = 60

# Initial Conditions computation
machine_buses_py = np.insert(pv_buses, 0, 0)
voltages_machines = vm_array_final[machine_buses_py] *np.exp(1j*va_array_final[machine_buses_py])
powers_machines_p = pg_array_final[machine_buses_py]
powers_machines_q = qg_array_final[machine_buses_py]
ini_conditions_comp = compute_initial_conditions(H, Rs, Xd, Xd_p, Xq, Xq_p, KA_vec, TA_vec, KE_vec, TE_vec, KF_vec, TF_vec)
ini_conditions_comp.upload_system_variables(voltages_machines, powers_machines_p, powers_machines_q) # TODO
ini_conditions_comp.compute_initial_conditions()
dif_vars_gen_ini, alg_vars_gen_ini, inputs_gen_ini = ini_conditions_comp.return_initial_conditions(sort_boolean=True)
ini_cond_simulation = np.concatenate([dif_vars_gen_ini, alg_vars_gen_ini])
voltage_angle_intercalats = np.ravel(np.column_stack((final_volt_magn, final_volt_angle)))
sim_variables = np.concatenate([ini_cond_simulation, voltage_angle_intercalats])

# Dynamic Models Machines
Vref = inputs_gen_ini
machine_model = Machine_Models(nominal_freq, dampings, H, Xd, Xd_p, Xq, Xq_p, Tdo_p, Tqo_p, Vref, KA_vec, TA_vec, KE_vec, TE_vec, KF_vec, TF_vec)
methods_map = machine_model.return_methods_mapping()

# Contingency Selection
fault_type = "load"
fault_time = 0.2
fault_magnitude = 0.08
fault_location = 7
contingency_lib, contingency_info = contingency_compiler(fault_type, fault_time, fault_magnitude, fault_location)

# Upload neural net
model_ml_sim = 'trained_PINN_1.pth'
pinn_weights, pinn_ranges = define_neural_net(model_ml_sim)
pinn_models_list = []

# Base DAE Solver
sim_time = 5.0
sim_time_step = 2.5e-2
assert sim_time > fault_time
tm_array_final = pg_array_final[machine_buses_py]
solver = Trapezoidal_dae_multitimestep(stator_ready_matrices, Y_adm, tm_array_final, machine_buses_py,
                                       sl_array_final, sim_variables, sim_time, sim_time_step, "trapezoidal")
solver.upload_dynamic_model(machine_model, "second", methods_map)
solver.upload_neural_nets(pinn_models_list, pinn_weights, pinn_ranges)
solver.apply_contingency(contingency_lib, contingency_info)
start_time = time.time()
time_array, states_array = solver.iterations_trapezoidal()
end_time = time.time()
simulation_time = end_time-start_time
print_sim_time_polished(simulation_time)

# Study Converged System
final_states = states_array[-1, :].flatten()
time_range_convergance = int(0.1/sim_time_step)
conv_boolean = convergence_solution(states_array[-time_range_convergance:, machine_buses_py.shape[0]*9:])

boolean_pinn = False
if len(pinn_models_list) > 0:
    boolean_pinn = True
# save_dynamic_simulation(time_array, states_array, (study_case, sim_time, sim_time_step, conv_boolean, pf_conv_boolean, simulation_time, boolean_pinn), (fault_type, fault_time, fault_magnitude, fault_location))

# Plotting methods
plotter = trajectories_overview(sim_time, (machine_buses_py, pg_buses), Y_adm, time_array, states_array)
plotter.show_states_machines()
plotter.show_voltages_evolution()
plotter.calc_power_flows_system()
plotter.calc_powers_machines(sl_array_final, contingency_info)
plotter.show_results()