#!/usr/bin/env python3
"""
amcvn_3d.py - Main script for running the radiative transfer solution to AM CVn.

15Jun18 - Abel Flores Prieto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter

from packages import general_formulas as gf
from packages import radiative_class as rc


def numbers():
    global n_rays
    while True:
        try:
            print("How many photon paths do you want to compute?")
            print("Or hit <Enter> for a default of 300.")
            n_rays = input("> ")
            if n_rays == '':
                n_rays = 300
                break
            else:
                n_rays = int(n_rays)
                assert n_rays >= 200
                break
        except:
            print("Please only input an integer above 200.")


def inclination_angle():
    global i
    while True:
        try:
            print("What inclination angle do you want? (from 0 to 180 degrees)")
            i = float(input("> "))
            assert i > 0 and i < 180
            break
        except:
            print("Angle is from 0 to 180 degrees non-inclusive.")


if __name__ == '__main__':
    # load the profile data
    txt_file = "2DRhoTVphi_AMCVNMdot_08732.txt"
    profile = np.loadtxt("data/" + txt_file)

    # load the line data
    hei = pd.read_csv("data/hei_lines.txt", delim_whitespace=True, skiprows=1)
    heii = pd.read_csv("data/heii_lines.txt", delim_whitespace=True, skiprows=1)

    # Create the frequency array
    nu_bb = []
    for freq1 in hei.nu_ik:
        nu_bb.append(gf.nu_peak(freq1))
    for freq2 in heii.nu_ik:
        nu_bb.append(gf.nu_peak(freq2))
    nu_bb = np.array(nu_bb).reshape(np.size(nu_bb))
    nu_gen = 10.**np.linspace(14, 16.3, num=200)

    nu = np.sort(np.append(nu_gen, nu_bb))  # fix frequencies

    # photo-ionization of HeI, which returns new frequency array as well
    sig_bf_I, nu = gf.sig_heI(nu)

    f1 = plt.figure(1)
    ax1 = f1.gca(projection='3d')

    print("Using profile data in {}".format(txt_file))
    print("""To use another profile data from a txt file, simply place it on the
data directory and change the variable txt_file in this file.
        """)

    print("""Each photon path has 50 points, if you want to change this, locate
the function parallel_lines3d() from the package general_functions in this file
and change the parameter num_path to the desired number of points.
        """)
    numbers()  # ask user for number of photon paths.
    inclination_angle()  # ask user for inclination angle.

    # start radiative object
    amcvn_3d = rc.Radiative3D(profile, hei, heii, nu, sig_bf_I)

    t0 = perf_counter()
    for x, y, z, n in gf.parallel_lines3d(n_rays, view=i, num_path=50):
        amcvn_3d.light_rays(x, y, z, n, bf_alpha=True, bb_alpha=True)
        print("\033[H\033[J")  # clears screen
        print("{} photon paths computed!".format(amcvn_3d.count))
        ax1.plot(x, y, z)
    t1 = perf_counter()

    print("\033[H\033[J")  # clears screen
    print("{} seconds to run {} photon paths.".format(t1 - t0, amcvn_3d.count))
    print("Average time for loop was {} s.".format((t1 - t0) / amcvn_3d.count))

    ax1.set_title("{} Photon Paths".format(amcvn_3d.count))
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_zlabel(r"$z$")
    f1.show()

    f2 = plt.figure(2)
    ax2 = f2.gca()
    ax2.plot(nu, amcvn_3d.I_nu)
    ax2.set_title(r'Frequency Spectrum - Inclination Angle $i = {:.1f}^\circ$'.format(i))
    ax2.set_xlabel(r'$\nu$ (Hz)')
    ax2.set_ylabel(r'$I_\nu$ (ergs cm$^{-2}$ s$^{-1}$ ster$^{-1}$ Hz$^{-1}$)')
    f2.show()

    f3 = plt.figure(3)
    ax3 = f3.gca()
    ax3.plot(nu, amcvn_3d.I_nu)
    ax3.set_title('Frequency Spectrum - Zoomed at Lines')
    ax3.set_xlabel(r'$\nu$ (Hz)')
    ax3.set_ylabel(r'$I_\nu$ (ergs cm$^{-2}$ s$^{-1}$ ster$^{-1}$ Hz$^{-1}$)')
    ax3.set_xlim([0.4e15, 0.8e15])  # lines
    f3.show()

    input("Press <Enter> to exit...")
