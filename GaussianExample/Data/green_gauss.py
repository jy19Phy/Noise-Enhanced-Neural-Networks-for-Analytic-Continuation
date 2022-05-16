# /usr/bin/env python
"""create imaginary time Green's function data for machine learing. """
import numpy as np
import scipy.integrate as integrate

# inverse temperature
BETA = 100.0
# number of positive/negative imaginary frequencies to be retained
OMEGA_POINTS_N = 5
# precision control
EPS = 1E-12


def gf_gauss(weight, sigma, mu, omega_n):
    """standardize integral to be mu=0, sigma=1 for evaluation.

    This will help to evalute very sharp/broad functions, and has performance gain.
    Refer to ./test/gauss_timing.py for benchmark.

    Notes
    -------
    For sharp peak we have to make the variable transformation by ourselves. QUADPACK
    just follow a routine to transform the integration range to be [0, 1], which
    evaluates to 0.0 for sharp peaks!!! Although we can split the integral, for
    radiculously sharp peaks we still have big problem---Besides many things, the
    constant factor 1/\sigma significantly amplify the error.
    """
    factor = 1.0 / np.sqrt(2.0*np.pi)
    real_f = lambda x: - (mu+x*sigma) / (omega_n**2 + (mu+x*sigma)**2) * np.exp(-x**2/2.0)
    imag_f = lambda x: - omega_n / (omega_n**2 + (mu+x*sigma)**2) * np.exp(-x**2/2.0)
    # ----------------------------------------------------------------------------------#
    # test program ./test/quad_singular_gauss.py programs shows 10 sigmas are enough for
    # all practical precision purposes (~ beta*10E-24). On the other hand, even with
    # standard (mu, sigma), whole space integral is not necessary the best numerically.
    # ----------------------------------------------------------------------------------#
    # The quad routine stops whenever epsabs or epsrel precision is achieved. Since they
    # have default value of 1E-8, to acquire more accurate result in terms of epsabs, we
    # need to set epsrel to zero.
    # ----------------------------------------------------------------------------------#
    real_g = integrate.quad(real_f, -10.0, 10.0, epsabs=EPS, epsrel=0)
    imag_g = integrate.quad(imag_f, -10.0, 10.0, epsabs=EPS, epsrel=0)
    real_g = (real_g[0]*factor, real_g[1]*factor)
    imag_g = (imag_g[0]*factor, imag_g[1]*factor)
    return (real_g, imag_g)


def test():
    """test several case against Mathematica calculation."""
    print("=========== testing function green_function_from_gaussian ===========")
    weight = 1.0
    sigma = 0.000018442565411526213
    mu = 4.633908371348811
    for n in range(-OMEGA_POINTS_N, OMEGA_POINTS_N):
        omega_n = (2*n + 1) * np.pi / BETA
        (real_g, imag_g) = gf_gauss(weight, sigma, mu, omega_n)
        print("2n+1 = {:3d}".format(2*n+1), " omega_n = {:6.3f}".format(omega_n),
              "re = {:20.16f}".format(real_g[0]), "  im =  {:20.16f}".format(imag_g[0]))
        print("2n+1 = {:3d}".format(2*n+1), "   errors        ",
              "re = {:20.16f}".format(real_g[1]), "  im =  {:20.16f}".format(imag_g[1]))


if __name__ == "__main__":
    test()
