#!/usr/bin/env python3
"""
short_characteristics.py - Numerical Method of Short Characteristics to solve
for the frequency spectrum of AM CVn's.

15Jun18 - Abel Flores Prieto
"""

import numpy as np


def linear_short(tau, S):
    '''
    Returns the outgoing and ingoing frequency spectra using the first-order
    (linear) Short Characteristics Method.
    '''
    si, nui = tau.shape
    I_ps, I_ms = np.zeros(tau.shape), np.zeros(tau.shape)

    I_ms[0, :] = np.zeros(nui)  # b.c. at end of disk

    d_tau_half = tau[1:, :] - tau[0:-1, :]
    x_half = 1. - np.exp(-d_tau_half)
    y_half = d_tau_half - x_half
    with np.errstate(all='ignore'):
        lamb = np.nan_to_num(y_half / d_tau_half)
        lamb_one = np.nan_to_num(-(y_half / d_tau_half) + x_half)

    for i in range(si - 1):
        # inward light rays
        I_ms[i + 1, :] = I_ms[i, :] * np.exp(-d_tau_half[i, :]) + lamb[i, :] * \
            S[i + 1, :] + lamb_one[i, :] * S[i, :]

    I_ps[-1, :] = I_ms[-1, :]  # b.c. at center of disk
    for i in range(si - 1)[::-1]:
        I_ps[i, :] = I_ps[i + 1, :] * np.exp(-d_tau_half[i, :]) + lamb[i, :] * \
            S[i, :] + lamb_one[i, :] * S[i + 1, :]

    return I_ps


def linear_short_notau(s, alpha, S):
    '''
    Returns the outgoing and ingoing frequency spectra using the first-order
    (linear) Short Characteristics Method.
    '''
    I_ps = np.zeros((len(s), len(alpha[0, :])))
    I_ms = np.zeros(I_ps.shape)

    si, nui = I_ps.shape

    I_ms[0, :] = np.zeros(nui)  # b.c. at end of disk

    d_tau_half = 0.5 * (alpha[1:, :][::-1] + alpha[0:-1, :][::-1]) * \
        (s[1:][::-1] - s[0:-1][::-1])[:, None]
    x_half = 1. - np.exp(-d_tau_half)
    y_half = d_tau_half - x_half
    with np.errstate(all='ignore'):
        lamb = np.nan_to_num(y_half / d_tau_half)
        lamb_one = np.nan_to_num(-(y_half / d_tau_half) + x_half)

    for i in range(si - 1):
        # inward light rays
        I_ms[i + 1, :] = I_ms[i, :] * np.exp(-d_tau_half[i, :]) + lamb[i, :] * \
            S[i + 1, :] + lamb_one[i, :] * S[i, :]

    I_ps[-1, :] = I_ms[-1, :]  # b.c. at center of disk
    for i in range(si - 1)[::-1]:
        I_ps[i, :] = I_ps[i + 1, :] * np.exp(-d_tau_half[i, :]) + lamb[i, :] * \
            S[i, :] + lamb_one[i, :] * S[i + 1, :]

    return I_ps
