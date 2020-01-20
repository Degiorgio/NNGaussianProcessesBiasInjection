# Author: K.Degiogio
#
# Utilities for loading a Dataframe and converting it into latex table

import ipdb
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.5f" % x)

model_ids = ["GP", "KANN", "KANN_extended"]


def print_norm():
    normal = "NORM/normal.pkl"
    df = pd.read_pickle(normal)
    print(df[model_ids].round(5).to_latex())


def print_cv():
    cv = "CROSSVAL/cross_validation_results.pkl"
    df = pd.read_pickle(cv)
    print(df[model_ids].round(5).to_latex())


def print_ab():
    ab1 = "AB/abs_wrong_prior.pkl"
    df1 = pd.read_pickle(ab1)[model_ids].T["Average"]
    ab2 = "AB/abs_noisy_kernel.pkl"
    df2 = pd.read_pickle(ab2)[model_ids].T["Average"]
    ab3 = "AB/abs_small_number_of_samples.pkl"
    df3 = pd.read_pickle(ab3)[model_ids].T["Average"]
    df = pd.concat([df1, df2, df3], axis=1).reset_index()
    df.columns = ["index", "incorrect prior", "noisy kernel", "small number of samples"]
    print(df.round(5).to_latex())


print("Norm------")
print_norm()
print("CV------")
print_cv()
print("AB------")
print_ab()
