import numpy as np
def return_knowns_guesses(system_info, system_ini_variables):
    slack_bus, pv_buses, pg_buses = system_info
    assert isinstance(slack_bus, int)
    assert slack_bus == 0
    no_sys_buses = 1 + pv_buses.shape[0] + pg_buses.shape[0]
    vm_array, va_array, pg_array, qg_array, pl_array, ql_array = system_ini_variables

    knowns = np.zeros((no_sys_buses, 2))
    guess_count = 0
    guesses = np.zeros(pv_buses.shape[0] + 2*pg_buses.shape[0])
    for i in range(no_sys_buses):
        if i == slack_bus:
            knowns[i][0] = vm_array[i]
            knowns[i][1] = va_array[i]
        elif i in pv_buses:
            knowns[i][0] = pg_array[i] - pl_array[i]
            knowns[i][1] = vm_array[i]
            guesses[guess_count] = va_array[0]
            guess_count += 1
        elif i in pg_buses:
            knowns[i][0] = pg_array[i] - pl_array[i]
            knowns[i][1] = qg_array[i] - ql_array[i]
            guesses[guess_count]   = vm_array[0]
            guesses[guess_count+1] = va_array[0]
            guess_count += 2
    if no_sys_buses in pv_buses:
        assert guess_count == pv_buses.shape[0] + 2*pg_buses.shape[0]
    elif no_sys_buses in pg_buses:
        assert guess_count == pv_buses.shape[0] + 2*pg_buses.shape[0] 

    return knowns, guesses