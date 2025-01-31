import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class trajectories_overview:
    def __init__(self, sim_time, sys_info, Y_adm, t_simulation, states_simulation) -> None:
        assert sim_time > 0
        assert t_simulation[-1] == sim_time
        self.simulation_range = sim_time
        self.machine_buses, self.pg_buses = sys_info
        self.no_gens = self.machine_buses.shape[0]
        self.t_sim_plot = t_simulation
        self.Y_adm = Y_adm
        self.no_buses = self.Y_adm.shape[0]
        self.establish_variables(states_simulation)

    def calc_angle_diff(self, angle1, angle2):
        diff = np.abs(angle1 - angle2)
        return np.minimum(diff, 2*np.pi - diff)

    def establish_variables(self, states_sim_plot):
        for idx, no_mac in enumerate(self.machine_buses):
            setattr(self, f"Eq_prime_{no_mac}", states_sim_plot[:, (idx)*7])
            setattr(self, f"Ed_prime_{no_mac}", states_sim_plot[:, (idx)*7+1])
            setattr(self, f"delta_{no_mac}", states_sim_plot[:, (idx)*7+2])
            setattr(self, f"omega_{no_mac}", states_sim_plot[:, (idx)*7+3])
            setattr(self, f"Efd_{no_mac}", states_sim_plot[:, (idx)*7+4])
            setattr(self, f"Rf_{no_mac}", states_sim_plot[:, (idx)*7+5])
            setattr(self, f"Vr_{no_mac}", states_sim_plot[:, (idx)*7+6])
            setattr(self, f"Id_{no_mac}", states_sim_plot[:, self.no_gens*7+idx*2])
            setattr(self, f"Iq_{no_mac}", states_sim_plot[:, self.no_gens*7+idx*2+1])
        for i in range(self.no_buses):
            setattr(self, f"voltage_magn_{i}", states_sim_plot[:, self.no_gens*9 + 2*i])
            setattr(self, f"voltage_angl_{i}", states_sim_plot[:, self.no_gens*9 + 2*i+1])

    def calculate_connections(self):
        dim_mat = self.Y_adm.shape[0]
        connections = dict()
        for i in range(dim_mat):
            array_inspect = self.Y_adm[i, i:]
            connection_indices = np.where(array_inspect != 0)[0]
            for ind in connection_indices:
                if i == ind+i:
                    continue
                elif i != ind+i:
                    connections[str(i)+'-'+str(ind+i)] = (i, ind+i)
        return connections

    def show_voltages_evolution(self):

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)
        
        ax0 = plt.subplot(gs[0, 0])
        for i in range(self.no_buses):
            ax0.plot(self.t_sim_plot, getattr(self, f"voltage_magn_{i}"), label=rf"$V_{{{i+1}}}$")
        ax0.legend()
        ax0.set_title('Voltage magnitude evolution')

        connections = self.calculate_connections()
        ax1 = plt.subplot(gs[0, 1])
        for i in connections.keys():
            ax1.plot(self.t_sim_plot, self.calc_angle_diff(getattr(self, f"voltage_angl_{connections[i][0]}"), getattr(self, f"voltage_angl_{connections[i][1]}")), 
                     label=rf"$\Theta_{{{connections[i][0]+1}}}-\Theta_{{{connections[i][1]+1}}}$")
        ax1.legend()
        ax1.set_title('\u0398 evolution')

    def show_states_machines(self):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(4, 2)
        
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])
        ax4 = plt.subplot(gs[2, 0])
        ax5 = plt.subplot(gs[2, 1])
        ax6 = plt.subplot(gs[3, 0])

        for i in self.machine_buses:
            ax0.plot(self.t_sim_plot, getattr(self, f"Eq_prime_{i}"), label=rf"$E'_{{q{i+1}}}$")
            ax1.plot(self.t_sim_plot, getattr(self, f"Ed_prime_{i}"), label=rf"$E'_{{d{i+1}}}$")
            ax2.plot(self.t_sim_plot, self.calc_angle_diff(getattr(self, f"delta_{i}"), getattr(self, f"voltage_angl_{i}")), label=rf"$\delta_{i+1}-\Theta_{i+1}$")
            ax3.plot(self.t_sim_plot, getattr(self, f"omega_{i}"), label=rf"$\omega_{{{i+1}}}$")
            ax4.plot(self.t_sim_plot, getattr(self, f"Efd_{i}"), label=rf"$Efd_{i+1}$")
            ax5.plot(self.t_sim_plot, getattr(self, f"Rf_{i}"), label=rf"$Rf_{i+1}$")
            ax6.plot(self.t_sim_plot, getattr(self, f"Vr_{i}"), label=rf"$Vr_{i+1}$")
        
        ax0.legend()
        ax0.set_title(r"$E'_{q}$ evolution")
        ax1.legend()
        ax1.set_title(r"$E'_{d}$ evolution")
        ax2.legend()
        ax2.set_title(r"$\delta$ evolution")
        ax3.legend()
        ax3.set_title(r"$\omega$ evolution")
        ax4.legend()
        ax4.set_title(r"$E_{fd}$ evolution")
        ax5.legend()
        ax5.set_title(r"$R_{fd}$ evolution")
        ax6.legend()
        ax6.set_title(r"$V_{r}$ evolution")

    
    def calc_powers_machines(self, power_demand, fault_info=None):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)

        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])

        p_array_total = 0
        for i in self.machine_buses:
            # Terminal power not taking into account the losses at the stator -- if there are any
            complex_voltage = getattr(self, f"voltage_magn_{i}") * np.exp(1j*getattr(self, f"voltage_angl_{i}"))
            complex_current_conj = getattr(self, f"Id_{i}")-1j*getattr(self, f"Iq_{i}")
            s_array = complex_voltage * complex_current_conj * np.exp(-1j*((getattr(self, f"delta_{i}") - np.pi/2)))
            p_array = np.real(s_array)
            p_array_total += p_array
            ax0.plot(self.t_sim_plot, p_array, label=rf"$P_{i+1}$")
            ax1.plot(self.t_sim_plot, getattr(self, f"omega_{i}"), label=rf"$f_{{{i+1}}}$")
        
        ax0.plot(self.t_sim_plot, p_array_total, label=r"$P_{total}$")
        total_load_power = np.ones_like(self.t_sim_plot)*np.sum(np.real(power_demand))
        if isinstance(fault_info, dict):
            fault_type = fault_info["fault_type"]
            fault_time = fault_info["fault_time"]
            fault_magn = fault_info["fault_magnitude"]
            fault_location = fault_info["fault_location"]
            pos = np.where(self.t_sim_plot == fault_time)[0][0]
            if fault_type == "load":
                total_load_power[pos+1:] = np.ones_like(self.t_sim_plot[pos+1:])*(np.sum(np.real(power_demand))+fault_magn)
            elif fault_type == "gen":
                pass
            elif fault_type == "none":
                pass
            else:
                raise Exception("The fault type does not exist.")
        ax0.plot(self.t_sim_plot, total_load_power, label=r"$P_{load}$")
        ax0.legend()
        ax0.set_title("Power output machines")
        ax1.legend()
        ax1.set_title("Freq machines")

    def calc_power_flows_system(self):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)
        connections = self.calculate_connections()
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])
        for i in connections.keys():
            bus_s_ind = connections[i][0]
            bus_r_ind = connections[i][1]
            power_r, power_s = self.calculate_power_flows(bus_s=bus_s_ind, bus_r=bus_r_ind)
            ax0.plot(self.t_sim_plot, np.real(power_r), 
                     label=rf"$S bus_{{{bus_s_ind+1}}}-R bus_{{{bus_r_ind+1}}}$")
            ax1.plot(self.t_sim_plot, np.imag(power_r), 
                     label=rf"$S bus_{{{bus_s_ind+1}}}-R bus_{{{bus_r_ind+1}}}$")
        ax0.legend()
        ax0.set_title('Active power flows')
        ax1.legend()
        ax1.set_title('Reactive power flows')

    def calculate_power_flows(self, bus_s, bus_r):
        voltage_s = getattr(self, f"voltage_magn_{bus_s}") * np.exp(1j*getattr(self, f"voltage_angl_{bus_s}"))
        voltage_r = getattr(self, f"voltage_magn_{bus_r}") * np.exp(1j*getattr(self, f"voltage_angl_{bus_r}"))
        voltage_dif = voltage_s - voltage_r
        admittance_value = np.abs(self.Y_adm[bus_s, bus_r])*np.exp(1j*np.angle(self.Y_adm[bus_s, bus_r]))
        power_r = voltage_r * np.conjugate(admittance_value * voltage_dif)
        power_s = voltage_s * np.conjugate(admittance_value * voltage_dif)
        return power_r, power_s

    def show_results(self, save_fig=False):
        for i, figure in enumerate(plt.get_fignums()):
            plt.figure(figure)
            plt.tight_layout()
            if save_fig:                               
                plt.savefig(f'overviewfinal_{i + 1}.png')
        plt.show()