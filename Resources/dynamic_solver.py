import torch
from torch.autograd.functional import jacobian



class Trapezoidal_dae_multitimestep():
    def __init__(self, stator_clear_matrices, Y_adm, pg_pf_re, gen_indices, pl_pf, ini_cond, t_final, step_size, integration_scheme):
        torch.set_default_dtype(torch.float64)
        self.t_final   = torch.tensor(t_final, dtype=torch.float64)
        self.step_size = torch.tensor(step_size, dtype=torch.float64)
        self.initial_state = torch.tensor(ini_cond, dtype=torch.float64)
        self.no_variables = self.initial_state.shape[0]
        self.stator_clear_matrices = torch.tensor(stator_clear_matrices, dtype=torch.float64)
        self.Yadmittance = torch.tensor(Y_adm, dtype=torch.complex128)
        self.no_buses_sys = self.Yadmittance.shape[0]
        self.pg_pf_re = torch.tensor(pg_pf_re, dtype=torch.float64)
        self.gen_indices = torch.tensor(gen_indices, dtype=torch.int)
        self.no_gens = self.gen_indices.shape[0]
        self.assign_vars_value()
        assert len(pl_pf) == self.no_buses_sys
        self.pl_pf = torch.tensor(pl_pf, dtype=torch.complex128)
        self.integration_scheme = integration_scheme
        self.cntgcy_sim = False

    def upload_neural_nets(self, ml_models, neural_net_weights, vars_ranges):
        dif_vars_ranges, alg_vars_ranges = vars_ranges
        self.ml_models_map = torch.tensor(ml_models, dtype=torch.int)
        assert set(self.ml_models_map.tolist()).issubset(set(self.gen_indices.tolist()))
        self.define_limits(dif_vars_ranges, alg_vars_ranges)
        self.neural_net_model = neural_net_weights
    
    def define_limits(self, dif_vars, alg_vars):
        self.eq_limits = dif_vars[0]
        self.ed_limits = dif_vars[1]
        self.delta_limits = dif_vars[2]
        self.omega_limits = dif_vars[3]

        self.Id0_limits = alg_vars[0]
        self.Id1_limits = alg_vars[1]
        self.Iq0_limits = alg_vars[2]
        self.Iq1_limits = alg_vars[3]

    def check_limits(self, dif_inputs, alg_inputs):
        value_eq_check, value_ed_check, value_delta_check, value_omega_check = dif_inputs
        value_id0_check, value_id1_check, value_iq0_check, value_iq1_check = alg_inputs
        if self.eq_limits[0] > value_eq_check or self.eq_limits[1] < value_eq_check:
            print('Eq_Careful', self.eq_limits, value_eq_check)
        if self.ed_limits[0] > value_ed_check or self.ed_limits[1] < value_ed_check:
            print('Ed_Careful', self.ed_limits, value_ed_check)
        if self.delta_limits[0] > value_delta_check or self.delta_limits[1] < value_delta_check:
            print('Delta_Careful', self.delta_limits, value_delta_check)
        if self.omega_limits[0] > value_omega_check or self.omega_limits[1] < value_omega_check:
            print('Omega_Careful', self.omega_limits, value_omega_check)

        if self.Id0_limits[0] > value_id0_check or self.Id0_limits[1] < value_id0_check:
            print('Id0_Careful', self.Id0_limits, value_id0_check)
        if self.Id1_limits[0] > value_id1_check or self.Id1_limits[1] < value_id1_check:
            print('Id1_Careful', self.Id1_limits, value_id1_check)
        if self.Iq0_limits[0] > value_iq0_check or self.Iq0_limits[1] < value_iq0_check:
            print('Iq0_Careful', self.Iq0_limits, value_iq0_check)
        if self.Iq1_limits[0] > value_iq1_check or self.Iq1_limits[1] < value_iq1_check:
            print('Iq1_Careful', self.Iq1_limits, value_iq1_check)


    def upload_dynamic_model(self, generic_machine_model, selected_model, models_mapping):
        if selected_model not in models_mapping:
            return generic_machine_model.fallback()
        method_name = models_mapping[selected_model]
        self.machine_function = getattr(generic_machine_model, method_name, None)
        if self.machine_function is None:
            return generic_machine_model.fallback()
        self.machine_class = generic_machine_model
        self.set_active_generation_machines()
    
    def set_active_generation_machines(self):
        self.machine_class.set_pg_pf(self.pg_pf_re)

    @torch.enable_grad()
    def trapezoidal_rule_func(self, ini_dif_states, next_dif_states, ini_alg_states, next_alg_states, func, no_machine) -> torch.Tensor:
        state_0 = ini_dif_states
        state_1 = next_dif_states
        alg_vars_0 = ini_alg_states
        alg_vars_1 = next_alg_states
        d_state_0 = func(state_0, alg_vars_0, no_machine)
        d_state_1 = func(state_1, alg_vars_1, no_machine)
        computation_trapz = state_1 - state_0 - 0.5*self.step_size*(d_state_0+d_state_1)
        return computation_trapz
    
    @torch.enable_grad()
    def backward_euler_func(self, ini_dif_states, next_dif_states, next_alg_states, func, no_machine) -> torch.Tensor:
        state_0 = ini_dif_states
        state_1 = next_dif_states
        alg_vars_1 = next_alg_states
        d_state_1 = func(state_1, alg_vars_1, no_machine)
        computation_be = state_1 - state_0 - self.step_size*d_state_1
        return computation_be

    def rk_integration_scheme(self, ini_dif_states, next_dif_states, ini_alg_states, next_alg_states, func, no_machine) -> torch.Tensor:
        if self.integration_scheme == 'trapezoidal':
            output_computation = self.trapezoidal_rule_func(ini_dif_states, next_dif_states, ini_alg_states, next_alg_states, func, no_machine)
        elif self.integration_scheme == 'backward_euler':
            output_computation = self.backward_euler_func(ini_dif_states, next_dif_states, next_alg_states, func, no_machine)
        return output_computation
    
    @torch.enable_grad()
    def pinn_integration_scheme(self, pinn_input) -> tuple:
        preds_pinn = self.neural_net_model(pinn_input)
        d_delta = preds_pinn[:, 0:1][0][0]
        d_omega = preds_pinn[:, 1:2][0][0]
        ctant_value = torch.tensor(0.)
        return torch.stack([ctant_value, ctant_value, d_delta, d_omega, ctant_value, ctant_value, ctant_value])
        # return torch.stack([d_delta, d_omega])
    
    def machine_stator_equations(self, next_dif_states, next_alg_states, no_machine):
        Eq_prime, Ed_prime, delta, *_ = next_dif_states
        I_d, I_q, Vm, Theta = next_alg_states
        voltage_vector = torch.stack([Ed_prime-Vm*torch.sin(delta-Theta),
                                      Eq_prime-Vm*torch.cos(delta-Theta)])

        stator_currents = torch.matmul(self.stator_clear_matrices[no_machine], voltage_vector)
        res_d = I_d - stator_currents[0]
        res_q = I_q - stator_currents[1]
        return torch.stack([res_d, res_q])
    
    def update_sys_angles(self, states):
        for i in range(self.no_gens):
            if states[i*7+2] > torch.pi:
                states[i*7+2] = states[i*7+2] - 2*torch.pi
            elif states[i*7+2] < -torch.pi:
                states[i*7+2] = states[i*7+2] + 2*torch.pi

        for i in range(self.no_buses_sys):
            if states[self.no_gens*9+i*2+1] > torch.pi:
                states[self.no_gens*9+i*2+1] = states[self.no_gens*9+i*2+1] - 2*torch.pi
            if states[self.no_gens*9+i*2+1] < -torch.pi:
                states[self.no_gens*9+i*2+1] = states[self.no_gens*9+i*2+1] + 2*torch.pi
        return states
    
    def assign_vars_value(self):
        self.dif_states_indices = torch.zeros((self.no_gens, 7), dtype=torch.int32)
        for i in range(self.no_gens):
            self.dif_states_indices[i] = torch.arange(i*7, i*7+7)

        self.alg_states_indices = torch.zeros((self.no_gens, 4), dtype=torch.int32)
        for i in range(self.no_gens):
            self.alg_states_indices[i] = torch.tensor([self.no_gens*7+i*2, self.no_gens*7+i*2+1, self.no_gens*9+self.gen_indices[i]*2, self.no_gens*9+self.gen_indices[i]*2+1])

        self.currents_d_axis = torch.zeros(self.no_gens, dtype=torch.int32)
        self.currents_q_axis = torch.zeros(self.no_gens, dtype=torch.int32)
        self.deltas_local_rs = torch.zeros(self.no_gens, dtype=torch.int32)
        for i in range(self.no_gens):
            self.currents_d_axis[i] = self.no_gens*7+i*2
            self.currents_q_axis[i] = self.no_gens*7+i*2+1
            self.deltas_local_rs[i] = i*7+2
        
        self.Vm_indices = torch.zeros(self.no_buses_sys, dtype=torch.int32)
        self.Theta_indices = torch.zeros(self.no_buses_sys, dtype=torch.int32)
        for i in range(self.no_buses_sys):
            self.Vm_indices[i] = self.no_gens*9 + i*2
            self.Theta_indices[i] = self.no_gens*9 + i*2 + 1

    def update_function_dev(self, states_x0, states_x1):
        results_final = torch.zeros(self.no_variables)
        for no_mac in range(self.no_gens):
            dif_var_ini_m = states_x0[self.dif_states_indices[no_mac]]
            alg_var_ini_m = states_x0[self.alg_states_indices[no_mac][:-1]]
            dif_var_new_m = states_x1[self.dif_states_indices[no_mac]]
            alg_var_new_m = states_x1[self.alg_states_indices[no_mac]]
            if no_mac in self.ml_models_map:
                dif_vars_check = dif_var_ini_m[0].item(), dif_var_ini_m[1].item(), dif_var_ini_m[2].item(), dif_var_ini_m[3].item()
                alg_vars_check = alg_var_ini_m[0].item(), alg_var_new_m[0].item(), alg_var_ini_m[1].item(), alg_var_new_m[1].item()
                self.check_limits(dif_vars_check, alg_vars_check)
                pinn_input_data = torch.cat([dif_var_ini_m[0].view(-1,1), dif_var_ini_m[1].view(-1,1), dif_var_ini_m[2].view(-1,1), dif_var_ini_m[3].view(-1,1),
                                             alg_var_ini_m[0].view(-1,1), alg_var_new_m[0].view(-1,1), alg_var_ini_m[1].view(-1,1), alg_var_new_m[1].view(-1,1), self.step_size.view(-1,1)], dim=1)
                output = self.pinn_integration_scheme(pinn_input_data)
                results_final[self.dif_states_indices[no_mac]] = dif_var_new_m - dif_var_ini_m - self.step_size*output
                results_final[self.alg_states_indices[no_mac][:2]] = self.machine_stator_equations(dif_var_new_m, alg_var_new_m, no_mac)
            else:
                results_final[self.dif_states_indices[no_mac]] = self.rk_integration_scheme(dif_var_ini_m, dif_var_new_m, alg_var_ini_m, alg_var_new_m, self.machine_function, no_mac)
                results_final[self.alg_states_indices[no_mac][:2]] = self.machine_stator_equations(dif_var_new_m, alg_var_new_m, no_mac)

        complex_vm = states_x1[self.Vm_indices] * torch.exp(1j * states_x1[self.Theta_indices])
        currents_net = torch.matmul(self.Yadmittance, complex_vm)
        generator_currents = (states_x1[self.currents_d_axis] + 1j * states_x1[self.currents_q_axis]) * torch.exp(1j * (states_x1[self.deltas_local_rs] - torch.pi / 2))
        reciprocal_vm = torch.reciprocal(torch.conj(complex_vm))
        load_currents      = torch.conj(self.pl_pf) * reciprocal_vm
        gen_currents = torch.zeros(self.no_buses_sys, dtype=torch.complex128)
        gen_currents[self.gen_indices]  = generator_currents
        total_currents = gen_currents - load_currents - currents_net

        results_final[self.Vm_indices] = torch.real(total_currents)
        results_final[self.Theta_indices] = torch.imag(total_currents)
        
        return results_final
    
    def update_function_base(self, states_x0, states_x1):
        results_final = torch.zeros(self.no_variables)
        for no_mac in range(self.no_gens):
            dif_var_ini_m = states_x0[self.dif_states_indices[no_mac]]
            alg_var_ini_m = states_x0[self.alg_states_indices[no_mac][:-1]]
            dif_var_new_m = states_x1[self.dif_states_indices[no_mac]]
            alg_var_new_m = states_x1[self.alg_states_indices[no_mac]]

            results_final[self.dif_states_indices[no_mac]] = self.rk_integration_scheme(dif_var_ini_m, dif_var_new_m, alg_var_ini_m, alg_var_new_m, self.machine_function, no_mac)
            results_final[self.alg_states_indices[no_mac][:2]] = self.machine_stator_equations(dif_var_new_m, alg_var_new_m, no_mac)

        complex_vm = states_x1[self.Vm_indices] * torch.exp(1j * states_x1[self.Theta_indices])
        currents_net = torch.matmul(self.Yadmittance, complex_vm)
        generator_currents = (states_x1[self.currents_d_axis] + 1j * states_x1[self.currents_q_axis]) * torch.exp(1j * (states_x1[self.deltas_local_rs] - torch.pi / 2))
        reciprocal_vm = torch.reciprocal(torch.conj(complex_vm))
        load_currents      = torch.conj(self.pl_pf) * reciprocal_vm
        gen_currents = torch.zeros(self.no_buses_sys, dtype=torch.complex128)
        gen_currents[self.gen_indices]  = generator_currents
        total_currents = gen_currents - load_currents - currents_net

        results_final[self.Vm_indices] = torch.real(total_currents)
        results_final[self.Theta_indices] = torch.imag(total_currents)
        
        return results_final

    def trapezoidal_method(self, states_x0, update_f):

        ## We could also apply a simple forward euler
        states_x1 = states_x0.detach().clone()
        it_count = 0

        residual_iteration = torch.ones(self.no_variables, dtype=torch.float64)
        num_max_iterations = 10
        tolerance_newton = torch.tensor(1e-8, dtype=torch.float64)

        def wrapped_update_function_base(states_x1):
            return self.update_function_base(states_x0, states_x1)
        
        def wrapped_update_function_dev(states_x1):
            return self.update_function_dev(states_x0, states_x1)

        while torch.max(torch.abs(residual_iteration)) > tolerance_newton and it_count < num_max_iterations:
            
            if update_f == 'base':
                states_x1.requires_grad_(True)
                # t1 = time.perf_counter(), time.process_time()
                jacobian_matrix = jacobian(wrapped_update_function_base, states_x1)
                # t2 = time.perf_counter(), time.process_time()
                # print(f" 1Real time: {t2[0] - t1[0]:.2f} seconds")
                # print(f" 1CPU time: {t2[1] - t1[1]:.2f} seconds")
                states_x1.requires_grad_(False)
                
                inverse_matrix = torch.linalg.inv(jacobian_matrix)
                residual_iteration = self.update_function_base(states_x0, states_x1)
            elif update_f == 'dev':
                states_x1.requires_grad_(True)
                # t1 = time.perf_counter(), time.process_time()
                jacobian_matrix = jacobian(wrapped_update_function_dev, states_x1)
                # t2 = time.perf_counter(), time.process_time()
                # print(f" 1Real time: {t2[0] - t1[0]:.2f} seconds")
                # print(f" 1CPU time: {t2[1] - t1[1]:.2f} seconds")
                states_x1.requires_grad_(False)
                
                inverse_matrix = torch.linalg.inv(jacobian_matrix)
                residual_iteration = self.update_function_dev(states_x0, states_x1)

            with torch.no_grad():
                x_increment = - torch.matmul(inverse_matrix, residual_iteration)
                states_x1 += x_increment

            it_count +=1

        states_x1 = self.update_sys_angles(states_x1)
            
        assert it_count < num_max_iterations

        return states_x1
    
    def apply_contingency(self, contingency_class, contingency_information):
        if contingency_class == None or contingency_information == None:
            return
        self.cntgcy_sim = True
        self.cntgcy_class = contingency_class()
        self.cntgcy_time = contingency_information["fault_time"]
        self.cntgcy_magn = contingency_information["fault_magnitude"]
        self.cntgcy_location = contingency_information["fault_location"]
        assert isinstance(self.cntgcy_class, object)
        assert isinstance(self.cntgcy_location, int) # for lines we need tuple as well

    def is_approx_integer(self, x, tol=1e-6):
        return abs(x - round(x)) < tol, int(x)
    
    def iterations_trapezoidal(self):
        no_steps_float = self.t_final/self.step_size
        approx_inter_boolean_1, no_steps_int = self.is_approx_integer(no_steps_float.item())
        assert approx_inter_boolean_1
        no_step_disturbance = None
        if self.cntgcy_sim == False:
            no_steps = no_steps_int + 1
            time_array_sim = torch.arange(no_steps)*self.step_size
            states_array_sim = torch.zeros((no_steps, self.no_variables), dtype=torch.float64)
        elif self.cntgcy_sim == True:
            no_step_disturbance_float = self.cntgcy_time/self.step_size
            approx_inter_boolean_2, no_step_disturbance = self.is_approx_integer(no_step_disturbance_float.item())
            assert approx_inter_boolean_2
            no_steps = no_steps_int + 2
            time_array_sim_ini = torch.arange(no_steps-1)*self.step_size
            states_array_sim = torch.zeros((no_steps, self.no_variables), dtype=torch.float64)
            time_array_sim = torch.cat((time_array_sim_ini[:no_step_disturbance + 1], 
                               time_array_sim_ini[no_step_disturbance:no_step_disturbance + 1], 
                               time_array_sim_ini[no_step_disturbance + 1:]))
        return self.run_algorithm(no_steps, time_array_sim, states_array_sim, no_step_disturbance)

    def run_algorithm(self, no_steps, time_array_sim, states_array_sim, no_step_disturbance):

        states_array_sim[0, :] = self.initial_state

        states_iteration = self.initial_state

        for i in range(1, no_steps):
            if self.cntgcy_sim:
                if i<=no_step_disturbance:
                    states_next_time_step = self.trapezoidal_method(states_iteration, 'base')
                elif i==no_step_disturbance+1:
                    self.cntgcy_class.update(self, ind=self.cntgcy_location, f_magnitude=self.cntgcy_magn)
                    og_step_size = self.step_size
                    self.step_size = torch.tensor(0.)
                    states_next_time_step = self.trapezoidal_method(states_iteration, 'base')
                    self.step_size = og_step_size
                elif i> no_step_disturbance+1:
                    states_next_time_step = self.trapezoidal_method(states_iteration, 'dev')
            else:
                states_next_time_step = self.trapezoidal_method(states_iteration, 'base')

            states_array_sim[i, :] = states_next_time_step

            states_iteration = states_next_time_step

        return time_array_sim.numpy(), states_array_sim.numpy()