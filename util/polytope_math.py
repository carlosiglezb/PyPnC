import numpy as np


def extract_plane_eqn_from_coeffs(coeffs: list):
    r"""
    Extract plane equations from halfspace coefficients
        :param coeffs: list of dictionaries containing halfspace coefficients
        a, b, c, d in the equation ax + by + cz + d = 0
    """
    H = np.zeros((len(coeffs), 3))
    d_vec = np.zeros((len(coeffs),))
    i = 0
    for h in coeffs:
        H[i] = np.array([h['a'], h['b'], h['c']])
        d_vec[i] = h['d']
        i += 1
    return H, d_vec
