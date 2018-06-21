#!/usr/bin/env python3
"""
radiative_class.py - Class for radiative transfer of AM CVn's.

15Jun18 - Abel Flores Prieto
"""

import numpy as np
from scipy.interpolate import interp2d
from . import general_formulas as gf
from . import short_characteristics as sc
from . import constants as cs


class Radiative3D():
    """ Radiative Transfer Solver for 3D Simulation Data

    Numerical implementation to radiative transfer of AM CVn's.
    """

    def __init__(self,
                 data,          # AM CVn profile data
                 hei_df,        # data frame for HeI data
                 heii_df,       # data frame for HeII data
                 frequency,     # frequency array for whole simulation
                 sig_I,         # cross section of HeI
                 shape=512,     # shape of square data
                 ):
        self.freq = frequency
        self.rho = interp2d(np.unique(data[:, 0]), np.unique(data[:, 1]),
                            data[:, 2].reshape((shape, shape)).T)
        self.tgas = interp2d(np.unique(data[:, 0]), np.unique(data[:, 1]),
                             data[:, 3].reshape((shape, shape)).T)
        self.vphi = interp2d(np.unique(data[:, 0]), np.unique(data[:, 1]),
                             data[:, 4].reshape((shape, shape)).T)
        self.sig_I = sig_I
        self.hei = hei_df
        self.heii = heii_df
        self.I_nu = np.zeros(len(frequency))
        self.count = 0

    def light_rays(self, x, y, z, n, bf_alpha=True, bb_alpha=True):
        """
        Adds the intensity of a given light ray to the spectrum variable.
        """
        r = np.sqrt(x**2 + y**2. + z**2.)
        theta = np.arccos(z / r)
        phi = np.arctan(y / x)
        s = np.sqrt((x - x[0])**2 + (y - y[0])**2. + (z - z[0])**2.)

        # calculate the new profiles, interp2d or RBS?
        rho = np.diagonal(self.rho(r, theta))
        tgas = np.diagonal(self.tgas(r, theta))
        vphi = np.diagonal(self.vphi(r, theta))

        # compute the number densities
        n_e, n_I, n_II, n_III = gf.num_density(rho, tgas)

        # relativistic gamma factor
        gamma = 1 / np.sqrt(1 - vphi**2 / cs.c**2)

        # absorption coefficients
        alpha = np.zeros((rho.shape[0], len(self.freq)))

        # free-free
        alpha += gf.alpha_ff(self.freq, tgas, n_e, n_II)
        alpha += gf.alpha_ff(self.freq, tgas, n_e, n_III)

        if bf_alpha:
            # bound-free
            for i in range(1, 20):
                alpha += gf.alpha_bf_II(self.freq, tgas, n_II, i)
            for gi, Ei in zip(self.hei.g_i, self.hei.E_i):
                alpha += gf.alpha_bf_I(self.freq, tgas, n_I, self.sig_I, gi, Ei)

        if bb_alpha:
            # bound-bound
            for nu_0, g_i, E_i, B_ik in zip(self.hei.nu_ik, self.hei.g_i,
                                            self.hei.E_i, self.hei.B_ik):
                phi_p = gf.thermal_broad(self.freq, tgas, nu_0)
                alpha += gf.alpha_bb(self.freq, tgas, n_I, 1., g_i, E_i, B_ik, phi_p)

            # only thermal broadening He II
            for nu_0, g_i, E_i, B_ik in zip(self.heii.nu_ik, self.heii.g_i,
                                            self.heii.E_i, self.heii.B_ik):
                phi_p = gf.thermal_broad(self.freq, tgas, nu_0)
                alpha += gf.alpha_bb(self.freq, tgas, n_II, 2., g_i, E_i, B_ik, phi_p)

        # transformation laws
        vec_vphi = np.zeros((len(vphi), 3))
        vec_vphi[:, 0] = -vphi * np.sin(phi)  # x-component of vector field in cart
        vec_vphi[:, 1] = vphi * np.cos(phi)  # y-component (no z-comp)

        trans_law = gamma * (1 - np.sum(n * vec_vphi, axis=1) / cs.c)
        alpha = trans_law[:, None] * alpha

        S_nu = gf.plancks_prec(self.freq, tgas)

        # finally add to the spectrum
        self.I_nu += sc.linear_short_notau(s, alpha, S_nu)[0, :]
        self.count += 1
