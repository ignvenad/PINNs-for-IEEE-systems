import numpy as np
import matplotlib.pyplot as plt
from Cases.file_define_bus_system import case_pglib_sys

class Calculate_Accuracy_class:
    def __init__(self, saved_case_base, saved_case_1, saved_case_2=None):
        matrix_folder = './Saved_sims/'
        data_base = np.load(matrix_folder+saved_case_base, allow_pickle=True)
        loaded_metadata_base = data_base['metadata'].item()
        self.time_array_base = data_base['matrix1']
        self.states_array_base = data_base['matrix2']
        base_step_size = loaded_metadata_base['time_step']
        self.time_contingency = loaded_metadata_base['fault_time']

        self.define_vars_to_check(saved_case_base)

        data_case_1 = np.load(matrix_folder+saved_case_1, allow_pickle=True)
        loaded_metadata_case_1 = data_case_1['metadata'].item()
        self.time_array_case_1 = data_case_1['matrix1']
        self.states_array_case_1 = data_case_1['matrix2']
        step_size_case_1 = loaded_metadata_case_1['time_step']
        assert loaded_metadata_case_1['fault_time'] == self.time_contingency
        ratio_time_steps_raw_case_1 = step_size_case_1/base_step_size
        boolean_ratio_case_1, self.ratio_time_step_case_1 = self.is_approx_integer(ratio_time_steps_raw_case_1)
        assert boolean_ratio_case_1

        self.double_case = False
        if isinstance(saved_case_2, str):
            self.double_case = True
            data_case_2 = np.load(matrix_folder+saved_case_2, allow_pickle=True)
            loaded_metadata_case_2 = data_case_2['metadata'].item()
            self.time_array_case_2 = data_case_2['matrix1']
            self.states_array_case_2 = data_case_2['matrix2']
            step_size_case_2 = loaded_metadata_case_2['time_step']
            assert loaded_metadata_case_1['fault_time'] == self.time_contingency
            ratio_time_steps_raw_case_2 = step_size_case_2/base_step_size
            boolean_ratio_case_2, self.ratio_time_step_case_2 = self.is_approx_integer(ratio_time_steps_raw_case_2)
            assert boolean_ratio_case_2
    
    def define_vars_to_check(self, string_case):

        if string_case[:6] == 'case9_':
            study_case_sys = case_pglib_sys('case9_ieee')
            slack_bus, pv_buses, pg_buses, _ = study_case_sys.define_buses_type()
        elif string_case[:6] == 'case14':
            study_case_sys = case_pglib_sys('case14_ieee')
            slack_bus, pv_buses, pg_buses, _ = study_case_sys.define_buses_type()
        elif string_case[:6] == 'case30':
            study_case_sys = case_pglib_sys('case30_ieee')
            slack_bus, pv_buses, pg_buses, _ = study_case_sys.define_buses_type()
        assert slack_bus == 0
        machine_buses_py = np.insert(pv_buses, 0, 0)
        no_gens  = machine_buses_py.shape[0]
        no_buses = machine_buses_py.shape[0] + pg_buses.shape[0]
        
        delta_strings = np.empty((no_gens,), dtype=object)
        omega_strings = np.empty((no_gens,), dtype=object)
        current_strings = np.empty((no_gens*2,), dtype=object)
        deltas_to_check = np.zeros(no_gens, dtype=np.int32)
        omegas_to_check = np.zeros(no_gens, dtype=np.int32)
        alg_states_to_check = np.zeros(no_gens*2, dtype=np.int32)
        self.partners_vars_dif   = np.zeros(no_gens,   dtype=np.int32)
        count_gen = 0
        for i in range(no_gens):
            deltas_to_check[i] = i*7 + 2
            omegas_to_check[i] = i*7 + 3
            alg_states_to_check[count_gen]   = no_gens*7 + i*2
            alg_states_to_check[count_gen+1] = no_gens*7 + i*2 + 1
            self.partners_vars_dif[i] = no_gens*9 + machine_buses_py[i]*2 + 1

            delta_strings[i] = f"$\delta_{{{machine_buses_py[i]+1}}}$"
            omega_strings[i] = f"$\omega_{{{machine_buses_py[i]+1}}}$"
            current_strings[count_gen] = f"$Id_{{{machine_buses_py[i]+1}}}$"
            current_strings[count_gen+1] = f"$Iq_{{{machine_buses_py[i]+1}}}$"
            count_gen += 2

        voltage_strings = np.empty((no_buses,), dtype=object)
        vm_currents_to_check = np.zeros(no_buses, dtype=np.int32) # no point in comparing absolute angles
        for i in range(no_buses):
            vm_currents_to_check[i] = no_gens*9 + i*2
            voltage_strings[i] = f"$Vm_{{{i+1}}}$"

        self.vars_radians_to_check = deltas_to_check
        assert self.vars_radians_to_check.shape[0] == self.partners_vars_dif.shape[0]
        self.variables_to_check = np.concatenate([omegas_to_check, alg_states_to_check, vm_currents_to_check])
        self.strings_display = np.concatenate((delta_strings, omega_strings, current_strings, voltage_strings))

    def calculate_algorithm(self, time_array_case, states_case, ratio_case):
        first_pos, second_pos = self.calculate_first_second_pos(time_array_case)
        assert self.double_check_init_conditions(first_pos, second_pos, states_case, ratio_case) < 1e-8
        no_checks_evo = time_array_case[second_pos+1:].shape[0]

        accuracy_table_radians = np.zeros((no_checks_evo, self.vars_radians_to_check.shape[0]))
        count = 0
        for i in range(second_pos+1, time_array_case.shape[0]):
            angles_base_case = self.calc_angle_diff(self.states_array_base[i*ratio_case-(ratio_case-1), self.vars_radians_to_check], self.states_array_base[i*ratio_case-(ratio_case-1), self.partners_vars_dif])
            angles_study_case = self.calc_angle_diff(states_case[i, self.vars_radians_to_check], states_case[i, self.partners_vars_dif])
            accuracy_table_radians[count, :] = angles_study_case - angles_base_case
            count += 1

        accuracy_table = np.zeros((no_checks_evo, self.variables_to_check.shape[0]))
        time_array_to_check = np.zeros(no_checks_evo)
        count = 0
        for i in range(second_pos+1, time_array_case.shape[0]):
            time_array_to_check[count] = time_array_case[i]
            accuracy_table[count, :] = states_case[i, self.variables_to_check] - self.states_array_base[i*ratio_case-(ratio_case-1), self.variables_to_check]
            count += 1

        assert np.array_equal(time_array_to_check[-10:], time_array_case[-10:])
        assert count == no_checks_evo

        accuracy_final = np.hstack((accuracy_table_radians, accuracy_table))
        return accuracy_final

    def calculate_first_second_pos(self, time_array_case):
        positions_cont = np.where(time_array_case == self.time_contingency)[0]
        assert positions_cont.shape[0] == 2
        assert positions_cont[1]- positions_cont[0] == 1
        pos_first_cont = positions_cont[0]
        pos_second_cont = positions_cont[1]
        return pos_first_cont, pos_second_cont
    
    def double_check_init_conditions(self, first_pos, second_pos, states_case, ratio_case):
        dif_init_cond = states_case[second_pos, self.variables_to_check] - self.states_array_base[first_pos*ratio_case+1, self.variables_to_check]
        return np.max(np.abs(dif_init_cond), axis=0)

    def is_approx_integer(self, x, tol=1e-6):
        return abs(x - round(x)) < tol, int(x)
    
    def calc_angle_diff(self, angle1, angle2):
        diff = np.abs(angle1 - angle2)
        return np.minimum(diff, 2*np.pi - diff)
    
    def create_arrays_tikz(self, table, case):
        if case == 1:
            first_pos, second_pos = self.calculate_first_second_pos(self.time_array_case_1)
        elif case == 2:
            first_pos, second_pos = self.calculate_first_second_pos(self.time_array_case_2)
        plotting_array = np.zeros(self.time_array_case_1.shape[0])
        count = 0
        for i in range(second_pos+1, self.time_array_case_1.shape[0]):
            plotting_array[i] = table[count, 7]
            count +=1
        plt.plot(self.time_array_case_1[:], plotting_array, label='tr')
        plt.legend()
        plt.show()

    
    def compare_tables(self, table_1, table_2):
        # self.create_arrays_tikz(table_2, 2)
        means_table_1 = np.mean(np.abs(table_1[:]), axis=0)
        means_table_2 = np.mean(np.abs(table_2[:]), axis=0)
        improv_booleans = means_table_1 > means_table_2
        improvement_percentage = np.abs(means_table_1 - means_table_2) / means_table_1 * 100

        assert means_table_1.shape[0] == self.strings_display.shape[0]
        assert means_table_2.shape[0] == self.strings_display.shape[0]

        return improv_booleans, improvement_percentage, self.strings_display
    
    def run_study(self):
        table_case_1 = self.calculate_algorithm(self.time_array_case_1, self.states_array_case_1, self.ratio_time_step_case_1)
        
        if self.double_case:
            table_case_2 = self.calculate_algorithm(self.time_array_case_2, self.states_array_case_2, self.ratio_time_step_case_2)
            return table_case_1, table_case_2
        
        return table_case_1

true_case   = 'case14_ieee_true_load14_0.149.npz'
traditional_case  = 'case14_ieee_traditional_load14_0.149.npz'
hybrid_case = 'case14_ieee_hybrid_load14_0.149.npz'

computing_accuracy = Calculate_Accuracy_class(true_case, traditional_case, hybrid_case)

table_normal, table_ml = computing_accuracy.run_study()
table_results, impro_percentage, strings_display = computing_accuracy.compare_tables(table_normal, table_ml)

for i in range(int(len(strings_display)/6)):
    strings = strings_display[i*6:i*6+6]
    impro = impro_percentage[i*6:i*6+6]
    interleaved = [item for pair in zip(strings, impro) for item in pair]

    for j in range(0, len(interleaved), 12):  # 12 because each batch contains 6 pairs (12 items)
        batch = interleaved[j:j+12]
        line = "|"
        for k in range(0, len(batch), 2):  # Print each pair
            s, imp = batch[k], batch[k+1]
            line += f"{s:<10} | {imp:<10.2f}|"
        print(line)