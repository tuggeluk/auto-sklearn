"""
Scrape OpenMl for  top runs on each dataset
"""

import openml
import os
import pandas as pa
import numpy as np
import pickle
openml.config.cache_directory = os.path.expanduser('cache_data')

# open and concatenate pickle files
print("os listdir")
cache_dir = openml.config.cache_directory
metadata_pickles = [x for x in os.listdir(cache_dir) if '.pickle' in x]


openml_metadata = dict()
openml_metadata["classification"] = dict()
openml_metadata["regression"] = dict()
openml_metadata["lc_prediction"] = dict()


for pickle_name in metadata_pickles:
    with open(os.path.join(cache_dir,pickle_name), 'rb') as handle:
        loaded_dict = pickle.load(handle)

        openml_metadata["classification"] = {**openml_metadata["classification"], **loaded_dict["classification"]}
        openml_metadata["regression"] = {**openml_metadata["regression"], **loaded_dict["regression"]}
        openml_metadata["lc_prediction"] = {**openml_metadata["lc_prediction"], **loaded_dict["lc_prediction"]}



print("evaluate_metadata")