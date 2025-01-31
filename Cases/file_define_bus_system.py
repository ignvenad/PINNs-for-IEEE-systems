import numpy as np
import os

class case_pglib_sys:
    def __init__(self, sys_sel):
        self.study_case = sys_sel
        possible_systems = ["case9_ieee", "case14_ieee", "case30_ieee"]
        assert self.study_case in possible_systems
        if self.study_case == "case9_ieee":
            self.load_system_hardcoded()
            self.pf_double_check = False
        else:
            self.load_system()
            self.pf_double_check = True

    def to_complex(self, x):
        x = x.replace(" ", "").replace("im", "j")
        if x in ["0", "0.0", "0.0+0.0j", "0+0j"]:
            return 0.0
        return complex(x)

    def load_system_hardcoded(self):
        self.Y_adm = np.array([  [-17.361j, 0, 0, 17.361j, 0, 0, 0, 0, 0],
                        [0, -16j, 0, 0, 0, 0, 16j, 0, 0],
                        [0, 0, -17.065j, 0, 0, 0, 0, 0, 17.065j],
                        [17.361j, 0, 0, 3.307-39.309j, -1.365+11.604j, -1.942+10.511j, 0, 0, 0],
                        [0, 0, 0, -1.365+11.604j, 2.553-17.338j, 0, -1.188+5.975j, 0, 0],
                        [0, 0, 0, -1.942+10.511j, 0, 3.224-15.841j, 0, 0, -1.282+5.588j],
                        [0, 16j, 0, 0, -1.188+5.975j, 0, 2.805-35.446j, -1.617+13.698j, 0],
                        [0, 0, 0, 0, 0, 0, -1.617+13.698j, 2.772-23.303j, -1.155+9.784j],
                        [0, 0, 17.065j, 0, 0, -1.282+5.588j, 0, -1.155+9.784j, 2.437-32.154j]], dtype=np.complex128)
        self.slack_bus = 0
        self.pv_buses = np.array([1, 2])
        self.pg_buses = np.array([3, 4, 5, 6, 7, 8])
        self.vm_array = np.array([1.04, 1.025, 1.025, 1., 1., 1., 1., 1., 1.])
        self.va_array = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0])
        self.pg_array = np.array([1., 1.63, 0.85, 0., 0., 0., 0., 0., 0.])
        self.qg_array = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.pl_array = np.array([0, 0, 0, 0, 1.25, 0.9, 0, 1, 0])
        self.ql_array = np.array([0, 0, 0, 0, 0.5, 0.3, 0, 0.35, 0])

    def load_system(self):

        files_directory = os.path.join("./Csv_files/", self.study_case)
        file_root = os.path.join(files_directory, f"pglib_opf_{self.study_case}")
        file_Yadm = file_root + "_Yadm.csv"
        file_btype = file_root + "_btype.csv"
        file_vm    = file_root + "_vm.csv"
        file_va    = file_root + "_va.csv"
        file_pg    = file_root + "_pg.csv"
        file_qg    = file_root + "_qg.csv"
        file_pl    = file_root + "_pl.csv"
        file_ql    = file_root + "_ql.csv"

        Y_adm_raw = np.loadtxt(file_Yadm, delimiter=",", dtype=str)
        self.Y_adm = np.vectorize(self.to_complex)(Y_adm_raw)

        bus_type_array = np.loadtxt(file_btype, delimiter=",")
        self.vm_array       = np.loadtxt(file_vm, delimiter=",")
        self.va_array       = np.loadtxt(file_va, delimiter=",")
        self.pg_array       = np.loadtxt(file_pg, delimiter=",")
        self.qg_array       = np.loadtxt(file_qg, delimiter=",")
        self.pl_array       = np.loadtxt(file_pl, delimiter=",")
        self.ql_array       = np.loadtxt(file_ql, delimiter=",")
        assert bus_type_array[0] == 1
        assert np.where(bus_type_array == 1)[0].shape[0] == 1
        self.slack_bus = 0
        self.pv_buses = np.where(bus_type_array == 2)[0]
        self.pg_buses = np.where(bus_type_array == 3)[0]
        assert 1 + self.pv_buses.shape[0] + self.pg_buses.shape[0] == bus_type_array.shape[0]

    def define_buses_type(self):
        returning_arrays = self.vm_array, self.va_array, self.pg_array, self.qg_array, self.pl_array, self.ql_array
        return self.slack_bus, self.pv_buses, self.pg_buses, returning_arrays
    
    def return_admittance_matrix(self):
        return self.Y_adm
    
    def apply_same_machine_in_sys(self, vars_ini_arrays):
        vars_ini_arrays[2][1] = 0.85
        return vars_ini_arrays
    
    def return_machine_types(self):
        if self.study_case == "case9_ieee":
            return [0, 1, 2]
        elif self.study_case == "case14_ieee":
            return [3, 2, 4, 4, 4]
        elif self.study_case == "case30_ieee":
            return [3, 2, 4, 4, 4, 4]
    
    def verify_power_flow_results(self, pf_new_variables):
        if not self.pf_double_check:
            print("PF results for test system cannot be verified")
        else:
            final_active_p, final_reactive_p, final_volt_magn, final_volt_angle = pf_new_variables
            assert np.max(np.abs(final_volt_magn - self.vm_array)) < 1e-8
            assert np.max(np.abs(final_volt_angle - self.va_array)) < 1e-8
            assert np.max(np.abs(final_active_p - (self.pg_array - self.pl_array))) < 1e-8
            assert np.max(np.abs(final_reactive_p - (self.qg_array - self.ql_array))) < 1e-8
            print("PF results agree with PowerModels.jl")