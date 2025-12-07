from typing import Any, List, Dict
import numpy as np


def is_iterable(x: Any)->bool:
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False


# def sigmoid(x):
#   return 1 / (1 + np.exp(-x))
#
#
# def hermitian(A: np.ndarray)-> np.ndarray:
#     return A.conj().T
#
# def is_hermitian(A: np.ndarray)->bool:
#     return np.allclose(A, hermitian(A), atol=1e-15)
#
# # def mag_of_complex_vector(z: np.ndarray)->float:
# #     if z.dtype != complex: raise ValueError
# #
# #     return np.sqrt((np.abs(z)**2).sum())
#
#
# def degree2rad(val_degrees):
#     return val_degrees * np.pi/180.

def dBm_to_Watt(val_dBm):
    return np.power(10, (val_dBm/10 - 3)  )

def dBW_to_Watt(val_dBW):
    return np.power(10, val_dBW/10)


# def get_random_positions(num_positions, box_low, box_high, rng=None):
#     if rng is None: rng = np.random.default_rng()
#
#     positions = np.empty((num_positions, 3))
#     for i in range(num_positions):
#         positions[i,:] = rng.uniform(low=box_low, high=box_high)
#     return positions
#
#
def sample_gaussian_standard_normal(size=None, rng=None):
    if rng is None: rng = np.random.default_rng()

    betta = 1/np.sqrt(2) * (rng.normal(0, 1, size=size) + 1j * rng.normal(0, 1, size=size))
    return betta


def split_to_close_to_square_factors(x: int):
    n1 = int(np.floor(np.sqrt(x)))
    while n1 >= 1:
        if x % n1 == 0: break
        n1 -= 1
    n2 = x // n1
    return max(n1, n2), min(n1, n2)
    #return min(n1, n2), max(n1, n2)


# def list_of_dicts_to_dict_of_lists(l_d: List[Dict[Any, Any]])->Dict[Any, List[Any]]:
#     d_l = dict()
#
#     first_dict = l_d[0]
#     for key in first_dict.keys():
#         d_l[key] = []
#
#     for d in l_d:
#         if len(list(d.keys())) < len(list(d_l.keys())):
#             raise ValueError("Found dict with fewer number of keys")
#
#         for key, value in d.items():
#             try:
#                 if isinstance(value, np.ndarray):
#                     value = value.tolist()
#
#                 d_l[key].append(value)
#             except KeyError:
#                 pass
#                 #raise ValueError("Found dict with extra keys.")
#
#     return d_l
#
#
# def val2dBm(x):
#     return 10*np.log10(x) + 30
#
# def val2dB(x):
#     return 10*np.log10(x)
#
#
# def class_to_dict(cls):
#     result = {}
#     for key, value in cls.__dict__.items():
#         if not key.startswith('__') and not callable(value):
#             if isinstance(value, type):
#                 result[key] = class_to_dict(value)
#             else:
#                 result[key] = value
#     return result


def repeat_num_to_list_if_not_list_already(num_or_list, expected_len) -> List:

    if isinstance(num_or_list, list) or isinstance(num_or_list, tuple) or isinstance(num_or_list, np.ndarray):
        if len(num_or_list) != expected_len:
            raise ValueError
        else:
            mylist = num_or_list
    else:
        mylist = [num_or_list]*expected_len

    return mylist