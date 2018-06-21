#!/usr/bin/env python3
"""
general_functions.py - Useful functions to use when solving for the frequency
spectrum of AM CVn's.

15Jun18 - Abel Flores Prieto
"""

import numpy as np
from . import constants as cs
from scipy.special import wofz
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("error")


def parallel_lines3d(n_rays, view, num_path=50, R_in=4.7045e+08, R_out=1.0960e+10):
    """
    Yields the photon paths as x, y, z and their directional vector n.
    """
    num_R = int(n_rays / 30)
    i, Omega = -view * (np.pi / 180), 0
    for ro in np.linspace(1.01 * R_in, R_out, num_R, endpoint=False):
        for w in np.linspace(0, 2 * np.pi, 30, endpoint=False):
            nx, ny, nz = np.sin(i) * np.sin(Omega), -np.sin(i) * np.cos(Omega), np.cos(i)
            rox = ro * (-np.sin(w) * np.cos(i) * np.sin(Omega) + np.cos(w) * np.cos(Omega))
            roy = ro * (np.sin(w) * np.cos(i) * np.cos(Omega) + np.cos(w) * np.sin(Omega))
            roz = ro * np.sin(w) * np.sin(i)
            s_lim = np.sqrt(R_out**2 - ro**2)
            if ro < R_in:
                s = np.linspace(np.sqrt(R_in**2 - ro**2), s_lim, num=num_path)
            else:
                s = np.linspace(-s_lim, s_lim, num=num_path)
            x, y, z = rox + s * nx, roy + s * ny, roz + s * nz
            n = np.array([nx, ny, nz])
            yield x, y, z, n


def nu_peak(freq_0, tgas_0=1e4, num=100):
    '''
    Returns frequencies around absorption line.
    '''
    del_freq_D = freq_0 / cs.c * np.sqrt(2. * cs.k * tgas_0 / cs.m_He)
    fwhm = 2. * np.sqrt(np.log(2)) * del_freq_D  # Full width at half maximum
    freq = np.linspace(freq_0 - fwhm, freq_0 + fwhm, num=num)
    return freq


def num_density(dens, tgas):
    '''
    Returns the number density of electron, HeI, HeII, and HeIII.
    '''
    c_I = 4. * (2. * np.pi * cs.m_e * cs.k * tgas / (cs.h**2.))**(3. / 2.) * \
        np.exp(-cs.kai_I / (cs.k * tgas))
    c_II = (2. * np.pi * cs.m_e * cs.k * tgas / (cs.h**2.))**(3. / 2.) * \
        np.exp(-cs.kai_II / (cs.k * tgas))
    n_tot = dens / cs.m_He

    a = 3.
    b = 3. * c_I
    c = 3. * c_I * c_II - c_I * n_tot
    d = -2. * c_I * c_II * n_tot

    n_e = np.zeros(len(tgas))
    for i in range(len(tgas)):
        r = np.roots([a, b[i], c[i], d[i]])
        for root in r:
            if root.imag == 0. and root.real > 0.:
                n_e[i] = root.real

    n_III = ((c_I * c_II) / (n_e**2. - c_I * c_II)) * (n_tot - 3. * n_e)
    n_II = 3. * n_e - 2. * n_III
    n_I = n_tot + n_III - 3. * n_e

    return n_e, n_I, n_II, n_III


def plancks_law(freq, tgas):
    '''
    Returns Planck's Law as a function of tgas and frequency, where the each row
    is for a specific tgas (from last element to first element) and each column
    for a specific frequency.
    '''
    B_nu = ((2 * cs.h * freq**3) / (cs.c**2)) * \
        (1 / (np.exp(cs.h * freq / (cs.k * tgas[::-1][:, None])) - 1))
    return B_nu


def plancks_prec(freq, tgas):
    '''
    Returns Planck's Law as a function of tgas and frequency, where the each row
    is for a specific tgas (from last element to first element) and each column
    for a specific frequency. Checks for overflows.
    '''
    with np.errstate(all='ignore'):
        check = np.nan_to_num(np.exp(cs.h * freq / (cs.k * tgas[::-1][:, None])))
    B_nu = ((2 * cs.h * freq**3) / (cs.c**2)) * (1 / (check - 1))
    return B_nu


def tau(s, alpha):
    '''
    Returns the optical depth as a function of distance (row) and frequency
    (column), from 0 to tau_max.
    '''
    tau = np.zeros((len(s), len(alpha[0, :])))
    for i in range(1, len(s)):
        tau[-1 - i, :] = 0.5 * (alpha[-i, :] + alpha[-i - 1, :]) * \
            (s[-i] - s[-i - 1]) + tau[-i]
    return tau[::-1]


# Cross section
def sig_heI(freq):
    """
    Returns the cross sectional area of the photoionization of HeI along
    with its frequency.
    """
    data = np.loadtxt("data/he1.px.gd.txt", skiprows=41)
    sigma = data[:, 1] * 1.0e-18  # 1Mb = 1e-18 cm^2
    nu = data[:, 0] * cs.Ry / cs.h

    nu_for_inter = np.append(np.array(freq[0]), nu)
    sig = np.append(np.array([0.0]), sigma)
    inter_f = interp1d(nu_for_inter, sig)

    ret_nu = np.less(nu, freq[-1])
    new_nu = np.sort(np.append(freq, nu[ret_nu]))
    return inter_f(new_nu), new_nu


# Line broadenings

# the one I am using for the Radiative Class
def thermal_broad(freq, tgas, freq_0):
    '''
    Returns the thermal line profile function as a function of temperature (row)
    and frequency (column).
    '''
    del_freq_D = (freq_0 / cs.c) * np.sqrt(2. * cs.k * tgas / cs.m_He)
    phi_freq = np.exp(-(freq - freq_0)**2. / del_freq_D[:, None]**2.) / \
        (del_freq_D[:, None] * np.sqrt(np.pi))
    return phi_freq


def lorentz_broad(freq, tgas, freq_0, gamma):
    '''
    Returns the natural line profile function as a function of temperature (row)
    and frequency (column).
    '''
    factor = gamma / (4. * np.pi)
    phi_freq = factor / ((freq - freq_0)**2. + factor**2.)
    return np.tile(phi_freq, (len(tgas), 1))


def voight_broad(freq, tgas, freq_0, gamma=5.e8):
    '''
    Returns the Voigt line function at freq_0 with thermal and Lorentz profiles
    '''
    sigma_t = (freq_0 / cs.c) * np.sqrt(2. * cs.k * tgas / cs.m_He) / 2.
    sigma_l = (gamma / (4. * np.pi))
    x = (freq - freq_0)
    phi_freq = np.real(wofz(((x + 1.j) * sigma_l) /
                            (sigma_t[:, None] * np.sqrt(2)))) / (sigma_t[:, None] * np.sqrt(2. * np.pi))
    return phi_freq


# Absorption coefficients

# Free-Free
def alpha_ff(freq, tgas, n_e, n_he):
    '''
    Returns the absorption coefficient as a function of tgas (row) and freq
    (column).
    '''
    front = ((4. * cs.e**6.) / (3. * cs.m_e * cs.h * cs.c)) * \
        np.sqrt(2. * np.pi / (3. * cs.k * cs.m_e))  # around 3.7e8
    g_ff = 1.0  # Gaunt factor
    alpha_freq = front * tgas[:, None]**(-1. / 2.) * n_e[:, None] * n_he[:, None] * \
        freq**(-3.) * (1. - np.exp(-cs.h * freq / (cs.k * tgas[:, None]))) * g_ff
    return alpha_freq


# Bound-Free
def alpha_bf_I(freq, tgas, n_he, sig_he, g_i, E_i):
    '''
    Returns the bound-free absorption coefficients for He I.
    '''
    E_i = np.abs(E_i)
    freq_gone = E_i / cs.h

    sigma = np.zeros(len(freq))
    sig_allowed = np.greater(freq, freq_gone)
    sigma[sig_allowed] = sig_he[sig_allowed]

    dens_state = n_he * g_i * np.exp(-E_i / (cs.k * tgas)) / 1.
    alpha = dens_state[:, None] * sigma
    return alpha


def alpha_bf_II(freq, tgas, n_he, n_tran):
    '''
    Returns the bound-free absorption coefficient for up to n_tran transitions.
    This is only for HeII.
    '''
    gaunt_factor = 1.
    fine_const = cs.e**2. * 2. * np.pi / (cs.h * cs.c)
    g_i = 2. * n_tran**2.
    E_i = cs.m_e * cs.c**2. * 2.**2. * fine_const**2. / (2. * n_tran**2.)
    freq_gone = E_i / cs.h

    sigma = np.zeros(len(freq))
    sig_he = 512. * np.pi**7. * cs.m_e * cs.e**10. * 2.**4. * gaunt_factor / \
        (3. * np.sqrt(3.) * cs.c * cs.h**6. * n_tran**5. * (freq * 2. * np.pi)**3.)
    sig_allowed = np.greater(freq, freq_gone)
    sigma[sig_allowed] = sig_he[sig_allowed]

    dens_state = n_he * g_i * np.exp(-E_i / (cs.k * tgas)) / 2.
    alpha = dens_state[:, None] * sigma
    return alpha


# Bound-Bound
def alpha_bb(freq, tgas, n_he, g_0, g_i, E_i, B_ik, phi):
    '''
    Returns the bound-bound absorption coefficients for each absorption line.
    '''
    ener = np.abs(E_i)
    dens_state = n_he * g_i * np.exp(-ener / (cs.k * tgas)) / g_0
    alpha = (cs.h * freq / (4. * np.pi)) * dens_state[:, None] * B_ik * \
            (1. - np.exp(-cs.h * freq / (cs.k * tgas[:, None]))) * phi
    return alpha
