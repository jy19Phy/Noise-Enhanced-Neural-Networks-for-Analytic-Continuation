# /usr/bin/env python
"""Generate imaginary time Green's function data for machine learing. """
# import sys
from random import random
import numpy as np
from weights import seed_weights
from green_gauss import gf_gauss
# from data_lorentz import green_function_from_lorentz as gf_lorentz

# inverse temperature
BETA = 10.0
# number of positive/negative imaginary frequencies to be retained [-2*n+1, 2*n-1]
OMEGA_POINTS_N = 10
# number of Gaussian/Lorentz samples
SAMPLE_N = 10000
# maximum number of peaks, smaller number of peaks generation is favored
PEAK_N = 10
# weights_number = int(xi**BIAS_PEAK_N  * PEAK_N + 1), BIAS_PEAK_N > 1 means samples
# with smaller number of peaks has more generations
BIAS_PEAK_N = 1.5
# minimum and maximum positions of the the Gaussian peaks
OMEGA_MIN = -10.0
OMEGA_MAX = 10.0
# NOISE LEVEL --- no noise in current script
# NOISE_SIGMA = 0.0

def help():
    print("call gauss_exact() to generate {:d} samples".format(SAMPLE_N))


def gauss_exact():
    with open("data_gauss_exact.dat", "w+",  buffering=1000000) as file:
        """data format.
           peakN
           weight1 sigma1 mu1  weight2 sigma2 mu2 ...
           real_g(-OMEGA_POINTS_N) imag_g(-OMEGA_POINTS_N) ...
           peakN
           weight1 sigma1 mu1  weight2 sigma2 mu2 ...
           real_g(-OMEGA_POINTS_N) imag_g(-OMEGA_POINTS_N) ...
           ...
           ...
        """
        for isample in range(SAMPLE_N):
            xi = random()
            peakN = int(xi**BIAS_PEAK_N * PEAK_N + 1)
            weightList = seed_weights(peakN)
            # simgaList = [OMEGA_MAX * random() * 0.5 for i in range(peakN)]
            # muList = [OMEGA_MAX * (2*random()-1) * 0.5 for i in range(peakN)]
            sigmaList = []
            muList = []
            for i in range(peakN):
                # smaller sigma is prefered (since more chanlleging),
                #
                #        |               |               |
                #        +-------------------------------+
                # -OMEGA_MAX      |            |        OMEGA_MAX
                #              range of mu
                #  With this setting of mu and sigma, A(omega) will at least start to decay
                #  beyond the [-OMEGA_MAX, OMEGA_MAX] interval
                xi = random()
                sigmaList.append(OMEGA_MAX * xi**2 * 0.5)
                muList.append(OMEGA_MAX * (2*random()-1) * 0.5)
            file.write(str(peakN) + '\n')
            for i in range(peakN):
                if i != peakN - 1:
                    file.write(str(weightList[i]) + '\t' +
                               str(sigmaList[i]) + '\t' + str(muList[i]) + '\t')
                else:
                    file.write(str(weightList[i]) + '\t' +
                               str(sigmaList[i]) + '\t' + str(muList[i]) + '\n')
            for n in range(-OMEGA_POINTS_N, OMEGA_POINTS_N):
                omega_n = (2*n+1)*np.pi/BETA
                real_g_sum = 0.0
                imag_g_sum = 0.0
                for i in range(peakN):
                    weight = weightList[i]
                    sigma = sigmaList[i]
                    mu = muList[i]
                    (real_g, imag_g) = gf_gauss(weight, sigma, mu, omega_n)
                    real_g_sum += real_g[0]
                    imag_g_sum += imag_g[0]
                if n != OMEGA_POINTS_N - 1:
                    file.write(str(real_g_sum) + '\t' + str(imag_g_sum) + '\t')
                else:
                    file.write(str(real_g_sum) + '\t' + str(imag_g_sum) + '\n')


if __name__ == "__main__":
    print("================= Instruction for using this program =================")
    help()
    gauss_exact()

