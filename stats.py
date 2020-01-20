# Author: K.Degiorgio
# utilities for statistical significance testing

import math
import pandas as pd
import numpy as np


def permutation_test(accA, accB, system_a, system_b, actual, R=5000):
    # system_a/b: (model_id, accuracy_score,
    #              results per instance,
    #              actual per instance)
    def swp(A, B):
        indexes = list(range(len(A)))
        index_perm = np.random.permutation(indexes)
        i = index_perm[: np.random.binomial(len(A), 0.5)]
        t = A[i]
        A[i] = B[i]
        B[i] = t
        return A, B

    adif = np.zeros(R)

    results_A = system_a.squeeze()
    results_B = system_b.squeeze()
    actual = actual.squeeze()

    assert len(results_A) == len(results_B) == len(actual)
    tacca = accA
    taccb = accB
    diff = np.abs(tacca - taccb)
    for i in range(R):
        sA, sB = swp(results_A, results_B)
        acc_A = (sA == actual).sum() / len(sA)
        acc_B = (sB == actual).sum() / len(sB)
        adif[i] = np.abs(acc_A - acc_B)
    p_value = ((adif >= diff).sum() + 1) / (R + 1)
    return p_value
