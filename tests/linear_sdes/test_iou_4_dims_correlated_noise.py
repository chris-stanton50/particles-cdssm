"""
Currently, there is an issue with the implementation of the transition density 
of the IntegratedOrnsteinUhlenbeck SDE: 

This issue occurs when the rough component is of higher than 1 dimension: (d_r >= 2),
and there are correlation effects in the driving noise of the rough components.

This file is a mimumum working example: this needs to be debugged in the future.
"""

import numpy as np
import particles_cdssm.sdes as sdes


def f_a(rho):
    # Convenience function for true values of _a
    return np.exp(-rho)

def f_a_1(rho):
    # Convenience function for true values of _a
    return (1/rho) * (1 - np.exp(-rho))


def f_b(rho, mu):
    # Convenience function for true values of _b
    return mu * (1 - np.exp(-rho))


def f_b_1(rho, mu):
    # Convenience function for true values of _b
    return (mu /rho) * (np.exp(-rho) + rho - 1) # t=1


def f_v(rho, phi_sq):
    # Convenience function for true values of _v
    return ((phi_sq)/ (2*rho)) * (1 - np.exp(-2*rho))


def g_v(rho_1, rho_2, phi_sq):
    # Convenience function for true values of _v
    return ((phi_sq)/ (rho_1 + rho_2)) * (1 - np.exp(-(rho_1+rho_2)))


def f_v_01(rho, phi_sq):
    # Convenience function for true values of _v
    a = ((phi_sq)/ (rho*rho)) * (1 - np.exp(-1.*rho))
    b = ((phi_sq)/ (2.*rho*rho)) * (1 - np.exp(-(2.*rho)))
    return a - b


def g_v_01(rho_1, rho_2, phi_sq):
    # Convenience function for true values of _v
    a = ((phi_sq)/ (rho_1 * rho_2)) * (1 - np.exp(-1.*rho_1))
    b = ((phi_sq)/ (rho_2*(rho_1 + rho_2))) * (1 - np.exp(-(rho_1+rho_2)))
    return a - b


def f_v_11(rho, phi_sq):
    # Convenience function for true values of _v
    a = ((phi_sq)/ (rho * rho))
    b_1 = 1.
    b_2 = (1. - np.exp(-1.*rho))/rho
    b_4 = (1. - np.exp(-2.*rho))/(2.*rho)
    return a *(b_1 - 2.*b_2 + b_4)


def g_v_11(rho_1, rho_2, phi_sq):
    # Convenience function for true values of _v
    a = ((phi_sq)/ (rho_1 * rho_2))
    b_1 = 1.
    b_2 = (1. - np.exp(-1.*rho_1))/rho_1
    b_3 = (1. - np.exp(-1.*rho_2))/rho_2
    b_4 = (1. - np.exp(-(rho_1+rho_2)))/(rho_1 + rho_2)
    return a *(b_1 - b_2 - b_3 + b_4)


sde_params = {'N': 1, 'dimX': 4, 'rho': np.array([[1., 2.]]), 'mu': np.array([[1., 2.]]), 'phi': np.array([[1., 0.], [0.9, np.sqrt(1.-0.9 ** 2)]])}
sde = sdes.IntegratedOrnsteinUhlenbeck(**sde_params)

test_value = sde._v(0., 1.)
true_value = np.array([[[f_v_11(1., 1.), g_v_11(1., 2., 0.9), f_v_01(1., 1.), g_v_01(1., 2., 0.9)], 
                        [g_v_11(2., 1., 0.9), f_v_11(2., 1.), g_v_01(2., 1., 0.9), f_v_01(2., 1.)], 
                        [f_v_01(1., 1.), g_v_01(2., 1., 0.9), f_v(1., 1.), g_v(1., 2., 0.9)],
                        [g_v_01(1., 2., 0.9), f_v_01(2., 1.), g_v(1., 2., 0.9), f_v(2., 1.)]]])

assert np.all(np.isclose(test_value, true_value))

# A = np.stack([np.diag([f_v(0.5, 1.), f_v(1.0, 1.)])] * 50 + [np.diag([f_v(0.5, 4.), f_v(1.0, 4.)])] * 50)

# print(A.shape)