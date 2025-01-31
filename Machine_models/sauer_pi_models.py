import torch

class Machine_Models:
    def __init__(self, sys_freq, mdamping, minertia, xd, xd_p, xq, xq_p, Tdo, Tqo, Vref, KA_vec, TA_vec, KE_vec, TE_vec, KF_vec, TF_vec):
        self.freq = sys_freq
        self.m_damping_vec = mdamping
        self.m_inertia_vec = minertia
        self.xd_vec = xd
        self.xdp_vec = xd_p
        self.xq_vec = xq
        self.xqp_vec = xq_p
        self.Tdo_vec = Tdo
        self.Tqo_vec = Tqo
        self.TA_vec = TA_vec
        self.TE_vec = TE_vec
        self.TF_vec = TF_vec
        self.KA_vec = KA_vec
        self.KE_vec = KE_vec
        self.KF_vec = KF_vec
        self.Vref_vec = Vref
        self.ctant_state = torch.tensor(0.)

    def return_methods_mapping(self):
        methods_mapping = {
            "second": "second_order_machine",
            "seventh": "seventh_order_machine"
            }
        return methods_mapping

    def set_pg_pf(self, pg_pf: torch.Tensor):
        self.pg_pf = pg_pf

    def second_order_machine(self, diff_states, alg_states, no_machine) -> torch.Tensor:
        Eq_prime, Ed_prime, delta, omega, Efd, Rf, VR = diff_states
        Id, Iq, *_ = alg_states
        d_Eq = self.ctant_state
        d_Ed = self.ctant_state
        d_delta = omega*2*torch.pi*self.freq
        d_omega = (self.pg_pf[no_machine] - Ed_prime * Id - Eq_prime * Iq - self.m_damping_vec[no_machine] * omega)/(2*self.m_inertia_vec[no_machine])
        d_Efd = self.ctant_state
        d_Rf  = self.ctant_state
        d_VR  = self.ctant_state
        return torch.stack([d_Eq, d_Ed, d_delta, d_omega, d_Efd, d_Rf, d_VR])
    
    def fourth_order_machine(self, diff_states, alg_states, no_machine) -> torch.Tensor:
        Eq_prime, Ed_prime, delta, omega, Efd, Rf, VR = diff_states
        Id, Iq, *_ = alg_states
        d_Eq = 1/self.Tdo_vec[no_machine] * ( - Eq_prime - (self.xd_vec[no_machine] - self.xdp_vec[no_machine]) * Id + Efd)
        d_Ed = 1/self.Tqo_vec[no_machine] * ( - Ed_prime + (self.xq_vec[no_machine] - self.xqp_vec[no_machine]) * Iq)
        d_delta = omega*2*torch.pi*self.freq
        d_omega = (self.pg_pf[no_machine] - Ed_prime * Id - Eq_prime * Iq - (self.xqp_vec[no_machine]-self.xdp_vec[no_machine])*Id*Iq - self.m_damping_vec[no_machine] * omega)/(2*self.m_inertia_vec[no_machine])
        d_Efd = self.ctant_state
        d_Rf  = self.ctant_state
        d_VR  = self.ctant_state
        return torch.stack([d_Eq, d_Ed, d_delta, d_omega, d_Efd, d_Rf, d_VR])
    
    def seventh_order_machine(self, diff_states, alg_states, no_machine) -> torch.Tensor:
        Eq_prime, Ed_prime, delta, omega, Efd, Rf, VR = diff_states
        Id, Iq, Vm, *_ = alg_states
        d_Eq = 1/self.Tdo_vec[no_machine] * ( - Eq_prime - (self.xd_vec[no_machine] - self.xdp_vec[no_machine]) * Id + Efd)
        d_Ed = 1/self.Tqo_vec[no_machine] * ( - Ed_prime + (self.xq_vec[no_machine] - self.xqp_vec[no_machine]) * Iq)
        d_delta = omega*2*torch.pi*self.freq
        d_omega = (self.pg_pf[no_machine] - Ed_prime * Id - Eq_prime * Iq - (self.xqp_vec[no_machine]-self.xdp_vec[no_machine])*Id*Iq - self.m_damping_vec[no_machine] * omega)/(2*self.m_inertia_vec[no_machine])
        d_Efd = 1/self.TE_vec[no_machine] * (-(self.KE_vec[no_machine] + 0.0039*torch.exp(1.555*Efd))*Efd + VR)
        d_Rf  = 1/self.TF_vec[no_machine] * (-Rf + self.KF_vec[no_machine]/self.TF_vec[no_machine]*Efd)
        d_VR  = 1/self.TA_vec[no_machine] * (-VR + self.KA_vec[no_machine] * Rf - self.KA_vec[no_machine]*self.KF_vec[no_machine]/self.TF_vec[no_machine]*Efd + self.KA_vec[no_machine]*(self.Vref_vec[no_machine] - Vm))
        return torch.stack([d_Eq, d_Ed, d_delta, d_omega, d_Efd, d_Rf, d_VR])
    
    def fallback(self):
        return "Unknown machine model"