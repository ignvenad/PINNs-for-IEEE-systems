import numpy as np

class compute_initial_conditions():

    def __init__(self, Hi, Rsi, Xd_i, Xd_pi, Xq_i, Xq_pi, KA_vec, TA_vec, KE_vec, TE_vec, KF_vec, TF_vec) -> None:
        self.no_machines = Hi.shape[0]
        self.H    = Hi
        self.Rs   = Rsi
        self.Xd   = Xd_i
        self.Xd_p = Xd_pi 
        self.Xq   = Xq_i
        self.Xq_p = Xq_pi
        self.TA_vec = TA_vec
        self.TE_vec = TE_vec
        self.TF_vec = TF_vec
        self.KA_vec = KA_vec
        self.KE_vec = KE_vec
        self.KF_vec = KF_vec

    def upload_system_variables(self, voltages, P_gen, Q_gen):
        assert self.no_machines == voltages.shape[0]
        assert self.no_machines == P_gen.shape[0]
        assert self.no_machines == Q_gen.shape[0]

        self.m_voltages = voltages
        self.p_gen      = P_gen
        self.q_gen      = Q_gen

    def current_network(self, P, Q) -> tuple:
        current_gen = np.conjugate(P+1j*Q)/np.conjugate(self.m_voltages)
        return np.real(current_gen), np.imag(current_gen)
    
    def compute_delta(self, ID, IQ) -> tuple:
        phasor = self.m_voltages + (self.Rs+1j*self.Xq)*(ID+1j*IQ)
        omega = np.zeros(self.no_machines)
        return np.angle(phasor, deg=False), omega

    def compute_local_magns(self, ID, IQ, delta) -> tuple:
        currents_gen = (ID+1j*IQ)*np.exp(-1j*(delta-np.pi/2))
        voltages_gen = self.m_voltages*np.exp(-1j*(delta-np.pi/2))
        return np.real(currents_gen), np.imag(currents_gen), np.real(voltages_gen), np.imag(voltages_gen)

    def compute_Ed_prime(self, Id, Iq, Vd) -> float:
        Ed_p = (self.Xq - self.Xq_p)*Iq
        Ed_p_ver = Vd + self.Rs*Id - self.Xq_p*Iq
        if np.mean(Ed_p-Ed_p_ver) < 0.01:
            return Ed_p
        else:
            print(Ed_p, Ed_p_ver)
            raise("Error computing Ed_prime. Verification not passed")

    def compute_Eq_prime(self, Id, Iq, Vq) -> float:
        return Vq+self.Rs*Iq+self.Xd_p*Id

    def compute_Efd(self, Id, Eq_p) -> float:
        return Eq_p + (self.Xd-self.Xd_p)*Id
    
    def check_mech_input(self, pm, eq, ed, id, iq):
        calculate_tm = ed*id + eq*iq + (self.Xq_p-self.Xd_p)*id*iq
        assert np.max(np.abs(calculate_tm-pm)) < 1e-8

    def calculate_avr_states(self, Efd, Vm):
        Vri = (self.KE_vec + 0.0039*np.exp(1.555*Efd))*Efd
        Rfi = self.KF_vec/self.TF_vec*Efd
        Vref = Vm + Vri/self.KA_vec
        return Vri, Rfi, Vref
    
    def compute_initial_conditions(self) -> list:
        IDgen, IQgen = self.current_network(self.p_gen, self.q_gen)
        deltas, omegas = self.compute_delta(IDgen, IQgen)
        Id_gen, Iq_gen, Vd_gen, Vq_gen = self.compute_local_magns(IDgen, IQgen, deltas)
        Ed_p = self.compute_Ed_prime(Id_gen, Iq_gen, Vd_gen)
        Eq_p = self.compute_Eq_prime(Id_gen, Iq_gen, Vq_gen)
        Efd = self.compute_Efd(Id_gen, Eq_p)
        self.check_mech_input(self.p_gen, Eq_p, Ed_p, Id_gen, Iq_gen)
        Vri, Rfi, Vref = self.calculate_avr_states(Efd, np.abs(Vd_gen+1j*Vq_gen))

        self.ini_dif_vars = (Eq_p, Ed_p, deltas, omegas, Efd, Rfi, Vri)
        self.ini_alg_vars = (Id_gen, Iq_gen)
        self.ini_inp_vars = Vref
    
    def return_initial_conditions(self, sort_boolean):
        if sort_boolean:
            Eq_p, Ed_p, deltas, omegas, Efd, Rfi, Vri = self.ini_dif_vars
            ini_dif_vars_sorted = np.zeros(self.no_machines*len(self.ini_dif_vars))
            for i in range(self.no_machines):
                ini_dif_vars_sorted[i*7:(i+1)*7] = [Eq_p[i], Ed_p[i], deltas[i], omegas[i], Efd[i], Rfi[i], Vri[i]]
            Id_gen, Iq_gen = self.ini_alg_vars
            ini_alg_vars_sorted = np.zeros(self.no_machines*len(self.ini_alg_vars))
            for i in range(self.no_machines):
                ini_alg_vars_sorted[i*2:(i+1)*2] = [Id_gen[i], Iq_gen[i]]
            return ini_dif_vars_sorted, ini_alg_vars_sorted, self.ini_inp_vars
        else:
            return self.ini_dif_vars, self.ini_alg_vars, self.ini_inp_vars