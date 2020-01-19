# Author: K.Degiorgio
# utilities for statistical significance testing

import math
import pandas as pd
import numpy as np

def sign_test(system_a, system_b, q=0.5):
    from scipy.stats import binom
    # system_a/b: (model_id, accuracy_score, results per instance)
    df_1 = pd.DataFrame.from_dict(system_a[2], orient='index')
    df_2 = pd.DataFrame.from_dict(system_b[2], orient='index')
    try:
        diff = (df_1 - df_2).replace(0, np.nan)
        counts = diff[0].value_counts(dropna=False)
        N = (2 * math.ceil(counts[np.nan] / 2)) + counts[1] + counts[-1]
        K = math.ceil(counts[np.nan] / 2) + min(counts[1], counts[-1])
        result = 0
        for i in range(0, K + 1):
            result += binom.pmf(n=N, p=q, k=i)
        pvalue = 2 * result
        return pvalue
    except:
        # TODO: temp ugly fix
        return 1


def permutation_test(system_a, system_b, R=5000):
    # system_a/b: (model_id, accuracy_score, 
    #              results per instance, 
    #              actual per instance)
    def swp(A, B):
        indexes = list(range(len(A)))
        index_perm = np.random.permutation(indexes)
        i = index_perm[:np.random.binomial(len(A), 0.5)]
        t = A[i]; A[i] = B[i]; B[i] = t
        return A, B
    adif = np.zeros(R)
    assert(system_a[3] == system_b[3])
    results_A = pd.DataFrame.from_dict(system_a[2], orient='index').to_numpy().squeeze()
    results_B = pd.DataFrame.from_dict(system_b[2], orient='index').to_numpy().squeeze()
    actual = pd.DataFrame.from_dict(system_a[3], orient='index').to_numpy().squeeze()
    assert(len(results_A) == len(results_B) == len(actual))
    tacca = system_a[1]
    taccb = system_b[1]
    diff = np.abs(tacca-taccb)
    for i in range(R):
        sA, sB = swp(results_A, results_B)
        acc_A = (sA == actual).sum()/len(sA)
        acc_B = (sB == actual).sum()/len(sB)
        adif[i] = np.abs(acc_A - acc_B)
    p_value = ((adif >= diff).sum()+1)/(R+1)
    return p_value


def pairwise_ss_test(model_ids, results, test=sign_test, verbose=True):
    df = pd.DataFrame(index=model_ids, columns=model_ids)
    for id_index, model_id  in enumerate(model_ids):
        for inner_id_index in  range(id_index+1):
            inner_model_id = model_ids[inner_id_index]
            if inner_model_id == model_id:
                df.loc[model_id].loc[model_id] = np.nan
                continue
            system_a = results[model_id]
            system_b = results[inner_model_id]
            p_value = round(test(system_a, system_b), 7)
            df.loc[model_id].loc[inner_model_id] = p_value
            if verbose:
                print(system_a, "vs", system_b, ": p-value:", p_value)
    return df
