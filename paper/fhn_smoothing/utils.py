import numpy as np

def obs_times_to_store(T):
    """
    Returns an array of 10 observation times and preprends time 1, giving a total of
    11 times.
    """
    Ts_to_store_excl_1 = np.linspace(T // 10, T, num=10, dtype=np.int64)
    arr_1 = np.array([1], dtype=np.int64)
    obs_times_to_store = np.concatenate([arr_1] + [Ts_to_store_excl_1], axis=0)
    obs_times_to_store = obs_times_to_store - 1 # subtract 1 from all times
    return obs_times_to_store