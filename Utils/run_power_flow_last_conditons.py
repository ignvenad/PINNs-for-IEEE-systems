import numpy as np

def convergence_solution(voltage_array):
    voltage_magnitudes = voltage_array[:, ::2]
    # voltage_angles_prev     = voltage_array[:, 1::2]
    # voltage_angles = calc_angle_diff(voltage_angles_prev, voltage_angles_prev[:, 0].reshape(-1, 1))
    # voltage_array = np.stack([voltage_magnitudes, voltage_angles], axis=1)
    maximum_values = np.max(voltage_magnitudes, axis=0)
    minimum_values = np.min(voltage_magnitudes, axis=0)
    if np.max(maximum_values-minimum_values) < 1e-4:
        print("The simulation converged to a final steady-state operating point.")
        return True
    else:
        print("Solution hasn't converged yet")
        return False
    
def load_power_flow(case):
    if case == "case9_ieee":
        study_case = "Case9"
    elif case == "case14_ieee":
        study_case = "Case14"
    elif case == "case30_ieee":
        study_case = "Case30"
        
    final_active_p = np.load(f"./Power_flow_results/{study_case}/final_active_p.npy")
    final_reactive_p = np.load(f"./Power_flow_results/{study_case}/final_reactive_p.npy")
    final_volt_magn = np.load(f"./Power_flow_results/{study_case}/final_volt_magn.npy")
    final_volt_angle = np.load(f"./Power_flow_results/{study_case}/final_volt_angle.npy")

    return final_active_p, final_reactive_p, final_volt_magn, final_volt_angle