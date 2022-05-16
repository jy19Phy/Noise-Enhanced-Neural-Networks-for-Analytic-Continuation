# /usr/bin/env python
"""create N random weights such that $sum_i weight_i = 1$."""
import sys
import random

# total number of Gaussian peaks
PEAK_N = 10

def seed_weights(weights_num):
    """create N random weights of the Gaussian peaks such that $sum_i weight_i = 1$.

    Basically we use (weights_num - 1) random numbers to divide the region [0, 1]
    into  weights_num random intervals. Each interval corresponds to weight_i.
    """
    # xi is a general name for random number, xis means a collection of xi
    # by initializing xis, the result is correct even weights_num = 1
    xis = [0.0, 1.0]
    for i in range(weights_num-1):
        xis.append(random.random())
    xis.sort()
    weightList = []
    for i in range(weights_num):
        weightList.append(xis[i+1] - xis[i])
    return weightList

if __name__ == "__main__":
    print("========= Test seed_weights(weights_num) =========")
    WEIGHTS = seed_weights(PEAK_N)
    if len(WEIGHTS) != PEAK_N:
        print("error in the size of created weights!")
        sys.exit(1)
    print(sum(WEIGHTS))
    WEIGHTS_ROUNDED = [round(elem, 2) for elem in WEIGHTS]
    print(WEIGHTS_ROUNDED)
