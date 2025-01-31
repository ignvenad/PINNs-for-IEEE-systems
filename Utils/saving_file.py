import numpy as np
from datetime import datetime

def save_dynamic_simulation(time_array, states_array, metadata, faultdata):
    case_study      = metadata[0]
    sim_time        = metadata[1]
    time_step       = metadata[2]
    conv_bool       = metadata[3]
    pf_conv_bool    = metadata[4]
    time_comp_sim   = metadata[5]
    boolean_ml      = metadata[6]
    fault_type      = faultdata[0]
    fault_time      = faultdata[1]
    fault_magnitude = faultdata[2]
    fault_location  = faultdata[3]

    metadata = {'case_study': case_study,
                'sim_time': sim_time,
                'time_step': time_step,
                'convergence_bool': conv_bool,
                'pf_final_bool': pf_conv_bool,
                'compute_sim_time': time_comp_sim,
                'fault_type': fault_type,
                'fault_time': fault_time,
                'fault_magnitude': fault_magnitude,
                'fault_location': fault_location,
                'creation_date': datetime.now().strftime('%d-%m-%Y')}

    if boolean_ml:
        np.savez(f'./Saved_sims/{case_study}_sim{sim_time}_ts{time_step}_ft_{fault_type}{fault_location}_{fault_magnitude}_ml.npz', matrix1=time_array, matrix2=states_array, metadata=metadata)
    else:
        np.savez(f'./Saved_sims/{case_study}_sim{sim_time}_ts{time_step}_ft_{fault_type}{fault_location}_{fault_magnitude}.npz', matrix1=time_array, matrix2=states_array, metadata=metadata)

if __name__ == "__main__":
    data = np.load('./Saved_sims/case30_ieee_sim20.0_ts0.001_ft_load19_0.08.npz', allow_pickle=True)
    loaded_metadata = data['metadata'].item()
    time_array = data['matrix1']
    states_array = data['matrix2']
    print(states_array.shape)
    print(loaded_metadata['creation_date'])
    print(loaded_metadata["compute_sim_time"]/60)
    print(loaded_metadata['fault_location'])
    print(loaded_metadata['fault_time'])