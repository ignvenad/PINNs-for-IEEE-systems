def print_sim_time_polished(sim_time):
    minutes = int(sim_time // 60)
    seconds = sim_time % 60
    print(f"Sim time: {minutes} mins and {seconds:.1f} secs")