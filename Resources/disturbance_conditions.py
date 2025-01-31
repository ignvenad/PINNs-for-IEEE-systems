from abc import ABC, abstractmethod
import sys

class Contingency(ABC):
    @abstractmethod
    def update(self, obj, **kwargs):
        pass

class Generation_Fault(Contingency):
    def update(self, obj, **kwargs):
        ind = kwargs.get("ind")
        f_magnitude = kwargs.get("f_magnitude")
        assert ind is not None, "Missing: 'ind'"
        assert isinstance(ind, int), f"'ind'--> int, got {type(ind).__name__}"
        assert f_magnitude is not None, "Missing: 'f_magnitude'"
        assert isinstance(f_magnitude, (float, int)), f"'f_magnitude' --> float or int, got {type(f_magnitude).__name__}"

        obj.pg_pf_re[ind] += float(f_magnitude)
        obj.machine_class.set_pg_pf(obj.pg_pf_re)

class Load_Fault(Contingency):
    def update(self, obj, **kwargs):
        ind = kwargs.get("ind")
        f_magnitude = kwargs.get("f_magnitude")
        assert ind is not None, "Missing: 'ind'"
        assert isinstance(ind, int), f"'ind'--> int, got {type(ind).__name__}"
        assert f_magnitude is not None, "Missing: 'f_magnitude'"
        assert isinstance(f_magnitude, (float, int)), f"'f_magnitude' --> float or int, got {type(f_magnitude).__name__}"

        obj.pl_pf[ind] += float(f_magnitude)

class Topology_Change(Contingency):
    def update(self, obj, **kwargs):
        ind_s = kwargs.get("ind_s")
        ind_r = kwargs.get("ind_r")
        assert ind_s is not None, "Missing: 'ind_s'"
        assert ind_r is not None, "Missing: 'ind_r'"
        assert isinstance(ind_s, int), f"'ind_s'--> int, got {type(ind_s).__name__}"
        assert isinstance(ind_r, int), f"'ind_r'--> int, got {type(ind_r).__name__}"

        line_admittance = obj.Yadmittance[ind_r, ind_s]
        obj.Yadmittance[ind_r, ind_s] = 0.
        obj.Yadmittance[ind_s, ind_r] = 0.
        obj.Yadmittance[ind_r, ind_r] += line_admittance
        obj.Yadmittance[ind_s, ind_s] += line_admittance

def contingency_compiler(type_f, time, magnitude, location):
    contingency_map = {"gen": "Generation_Fault", "load": "Load_Fault", "topology": "Topology_Change"}
    if type_f == "none":
        return None, None
    if type_f not in contingency_map:
        return "Fault type does not exist"
    module_name = sys.modules[__name__]
    contingency_class = getattr(module_name, contingency_map[type_f])
    contingency_info = {"fault_type": type_f, "fault_time": time, "fault_magnitude": magnitude, "fault_location": location}
    return contingency_class, contingency_info