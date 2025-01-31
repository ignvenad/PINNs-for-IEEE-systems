import numpy as np

class return_machine_parameters:
    def __init__(self, machine_types):
        self.no_machines = len(machine_types)
        self.H = np.zeros(self.no_machines)
        self.Rs = np.zeros(self.no_machines)
        self.Xd = np.zeros(self.no_machines)
        self.Xd_p = np.zeros(self.no_machines)
        self.Xq = np.zeros(self.no_machines)
        self.Xq_p = np.zeros(self.no_machines)
        self.Tdo_p = np.zeros(self.no_machines)
        self.Tqo_p = np.zeros(self.no_machines)
        self.damping = np.zeros(self.no_machines)
        self.KA = np.zeros(self.no_machines)
        self.TA = np.zeros(self.no_machines)
        self.KE = np.zeros(self.no_machines)
        self.TE = np.zeros(self.no_machines)
        self.KF = np.zeros(self.no_machines)
        self.TF = np.zeros(self.no_machines)
        self.machine_types = machine_types
        self.machine_parameters_calculated = False
    
    def return_machine_params(self, simplification_transient_saliency):
        for idx, elem in enumerate(self.machine_types):
            if elem == 0:
                params_machine = self.slow_machine()
            elif elem == 1:
                params_machine = self.medium_machine()
            elif elem == 2:
                params_machine = self.fast_machine()
            elif elem == 3:
                params_machine = self.sg_machine_1()
            elif elem == 4:
                params_machine = self.sync_machine_2()
            else:
                raise Exception("No machine params of this type")
            
            self.H[idx] = params_machine["H"]
            self.Rs[idx] = params_machine["Rs"]
            self.Xd[idx] = params_machine["Xd"]
            self.Xd_p[idx] = params_machine["Xd_p"]
            self.Xq[idx] = params_machine["Xq"]
            self.Xq_p[idx] = params_machine["Xq_p"]
            self.Tdo_p[idx] = params_machine["Tdo_p"]
            self.Tqo_p[idx] = params_machine["Tqo_p"]
            self.damping[idx] = params_machine["damping"]

        if simplification_transient_saliency:
            self.Xq_p = self.Xd_p
        
        self.machine_parameters_calculated = True
            
        return (self.H, self.Rs, self.Xd, self.Xd_p, self.Xq, self.Xq_p, self.Tdo_p, self.Tqo_p, self.damping)
    
    def return_stator_matrices(self):
        assert self.machine_parameters_calculated

        impedance_matrices = np.zeros((self.no_machines, 2, 2))
        for i in range(self.no_machines):
            stator_matrix = np.array([[self.Rs[i], -self.Xq_p[i]],
                                      [self.Xd_p[i], self.Rs[i]]], dtype=np.float64)
            
            ainv = np.linalg.inv(stator_matrix)
            impedance_matrices[i] = ainv

        return impedance_matrices

    def return_params_avr(self):
        for idx, elem in enumerate(self.machine_types):
            if elem == 0:
                params_machine = self.slow_machine()
            elif elem == 1:
                params_machine = self.medium_machine()
            elif elem == 2:
                params_machine = self.fast_machine()
            elif elem == 3:
                params_machine = self.sg_machine_1()
            elif elem == 4:
                params_machine = self.sync_machine_2()
            else:
                raise Exception("No machine params of this type")
            self.KA[idx] = params_machine["KA"]
            self.TA[idx] = params_machine["TA"]
            self.KE[idx] = params_machine["KE"]
            self.TE[idx] = params_machine["TE"]
            self.KF[idx] = params_machine["KF"]
            self.TF[idx] = params_machine["TF"]
            
        return (self.KA, self.TA, self.KE, self.TE, self.KF, self.TF)
    
    def slow_machine(self):
        # Power system dynamics and stability / Peter W. Sauer and M. A. Pai
        H = 23.64
        Rs = 0
        Xd = 0.146
        Xd_p = 0.0608
        Xq = 0.0969
        Xq_p = 0.0969
        Tdo_p = 8.96
        Tqo_p = 0.31
        damping = 2.364
        KA = 20
        TA = 0.2
        KE = 1
        TE = 0.314
        KF = 0.063
        TF = 0.35
        dict_params = {"H": H, "Rs": Rs, "Xd": Xd, "Xd_p": Xd_p, "Xq": Xq, "Xq_p": Xq_p, "Tdo_p": Tdo_p, "Tqo_p": Tqo_p,
                       "damping": damping, "KA": KA, "TA": TA, "KE": KE, "TE": TE, "KF": KF, "TF": TF}
        return dict_params

    def medium_machine(self):
        # Power system dynamics and stability / Peter W. Sauer and M. A. Pai
        H = 6.4
        Rs = 0.
        Xd = 0.8958
        Xd_p = 0.1198
        Xq = 0.8645
        Xq_p = 0.1969
        Tdo_p = 6.
        Tqo_p = 0.535
        damping = 1.28
        KA = 20.
        TA = 0.2
        KE = 1.
        TE = 0.314
        KF = 0.063
        TF = 0.35
        dict_params = {"H": H, "Rs": Rs, "Xd": Xd, "Xd_p": Xd_p, "Xq": Xq, "Xq_p": Xq_p, "Tdo_p": Tdo_p, "Tqo_p": Tqo_p,
                       "damping": damping, "KA": KA, "TA": TA, "KE": KE, "TE": TE, "KF": KF, "TF": TF}
        return dict_params

    def fast_machine(self):
        # Power system dynamics and stability / Peter W. Sauer and M. A. Pai
        H = 3.01
        Rs = 0
        Xd = 1.3125
        Xd_p = 0.1813
        Xq = 1.2578
        Xq_p = 0.25
        Tdo_p = 5.89
        Tqo_p = 0.6
        damping = 0.903
        KA = 20
        TA = 0.2
        KE = 1
        TE = 0.314
        KF = 0.063
        TF = 0.35
        dict_params = {"H": H, "Rs": Rs, "Xd": Xd, "Xd_p": Xd_p, "Xq": Xq, "Xq_p": Xq_p, "Tdo_p": Tdo_p, "Tqo_p": Tqo_p,
                       "damping": damping, "KA": KA, "TA": TA, "KE": KE, "TE": TE, "KF": KF, "TF": TF}
        return dict_params
    
    def sg_machine_1(self):
        # P.Demetriou et al., Dynamic ieee test systemsfor transient analysis,
        # IEEE Systems Journal, vol. 11, no. 4, 2017.
        H = 11.89
        Rs = 0
        Xd = 1.3125
        Xd_p = 0.1813
        Xq = 1.2578
        Xq_p = 0.25
        Tdo_p = 0.5871
        Tqo_p = 0.1351
        damping = 8.96
        KA = 20
        TA = 0.2
        KE = 1
        TE = 0.314
        KF = 0.063
        TF = 0.35
        dict_params = {"H": H, "Rs": Rs, "Xd": Xd, "Xd_p": Xd_p, "Xq": Xq, "Xq_p": Xq_p, "Tdo_p": Tdo_p, "Tqo_p": Tqo_p,
                       "damping": damping, "KA": KA, "TA": TA, "KE": KE, "TE": TE, "KF": KF, "TF": TF}
        return dict_params
        
    def sync_machine_2(self):
        # P.Demetriou et al., Dynamic ieee test systemsfor transient analysis,
        # IEEE Systems Journal, vol. 11, no. 4, 2017.
        H = 4.985
        Rs = 0
        Xd = 1.3125
        Xd_p = 0.1813
        Xq = 1.2578
        Xq_p = 0.25
        Tdo_p = 1.1
        Tqo_p = 0.1086
        damping = 2
        KA = 20
        TA = 0.2
        KE = 1
        TE = 0.314
        KF = 0.063
        TF = 0.35
        dict_params = {"H": H, "Rs": Rs, "Xd": Xd, "Xd_p": Xd_p, "Xq": Xq, "Xq_p": Xq_p, "Tdo_p": Tdo_p, "Tqo_p": Tqo_p,
                       "damping": damping, "KA": KA, "TA": TA, "KE": KE, "TE": TE, "KF": KF, "TF": TF}
        return dict_params
        
