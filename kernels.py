# Author: K.Degiorgio
#
import ipdb
import numpy as np
import sklearn.gaussian_process as gp


def kernel_rbf_with_noise(n=10, scaling_f=1, length_scale=0.2, noise_level=9.0):
    k = scaling_f ** 2 * gp.kernels.RBF(length_scale=length_scale)
    k += gp.kernels.WhiteKernel(noise_level=noise_level)
    parameters = {
        "kernel__k1__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k2__noise_level": np.linspace(0.1, 10, num=n),
        "kernel__k1__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return k, parameters


def kernel_rbf_with_no_noise(n=10, scaling_f=1.0, length_scale=1.0):
    k = scaling_f ** 2 * gp.kernels.RBF(length_scale=length_scale)
    parameters = {
        "kernel__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return k, parameters


def kernel_sigmoid_no_noise(n=10, length_scale=0.2, periodicity=10.0):
    K = 1.0 * gp.kernels.ExpSineSquared(
        length_scale=length_scale,
        periodicity=periodicity,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(1.0, 10.0),
    )
    parameters = {
        "kernel__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k2__periodicity": np.linspace(1.0, 10, num=n),
        "kernel__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return K, parameters


def kernel_sigmoid_no_noise2(n=10, length_scale=0.2, periodicity=1.0):
    K = 1.0 * gp.kernels.ExpSineSquared(
        length_scale=length_scale,
        periodicity=periodicity,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(1.0, 10.0),
    )
    parameters = {
        "kernel__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k2__periodicity": np.linspace(1.0, 10, num=n),
        "kernel__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return K, parameters


def kernel_sigmoid_no_noise3(n=10, length_scale=9, periodicity=1.0):
    K = 1.0 * gp.kernels.ExpSineSquared(
        length_scale=length_scale,
        periodicity=periodicity,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(1.0, 10.0),
    )
    parameters = {
        "kernel__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k2__periodicity": np.linspace(1.0, 10, num=n),
        "kernel__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return K, parameters


def kernel_sigmoid_no_noise4(n=10, length_scale=1, periodicity=0.4):
    K = 1.0 * gp.kernels.ExpSineSquared(
        length_scale=length_scale,
        periodicity=periodicity,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(0.1, 10.0),
    )
    parameters = {
        "kernel__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k2__periodicity": np.linspace(1.0, 10, num=n),
        "kernel__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return K, parameters


def kernel_sigmoid_no_noise5(n=10, length_scale=1, periodicity=0.1):
    K = 1.0 * gp.kernels.ExpSineSquared(
        length_scale=length_scale,
        periodicity=periodicity,
        length_scale_bounds=(0.1, 10.0),
        periodicity_bounds=(0.1, 10.0),
    )
    parameters = {
        "kernel__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k2__periodicity": np.linspace(1.0, 10, num=n),
        "kernel__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return K, parameters


def kernel_sigmoid_with_noise(
    n=10, scaling_f=1, length_scale=0.1, periodicity=0.1, noise_level=8
):
    K = scaling_f * gp.kernels.ExpSineSquared(
        length_scale=length_scale, periodicity=periodicity
    )
    K = K + gp.kernels.WhiteKernel(noise_level=noise_level)
    parameters = {
        "kernel__k1__k2__length_scale": np.linspace(0.1, 10, num=n),
        "kernel__k2__noise_level": np.linspace(0.1, 10, num=n),
        "kernel__k1__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return K, parameters


def kernel_dot_product(n=10, scaling_f=1.0, sigma_0=1.0):
    K = scaling_f * gp.kernels.DotProduct(sigma_0=sigma_0)
    parameters = {
        "kernel__k2__sigma_0": np.linspace(0, 10, num=n),
        "kernel__k1__constant_value": np.linspace(0.1, 10, num=n),
    }
    return K, parameters


def kernel_matern(n=10, scaling_f=1.0, nu=0.5):
    K = scaling_f * gp.kernels.Matern(nu=nu)
    parameters = {"kernel__k1__constant_value": np.linspace(0.1, 10, num=n)}
    return K, parameters
