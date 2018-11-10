"""
Basic usage of OpenMl
"""

import openml
import openmlkey
from sklearn import neighbors
import os
from scripts.update_metadata_util import classification_tasks_small as classification_tasks
from scripts.update_metadata_util import regression_tasks_small as regression_tasks
import pandas as pa
import pickle
openml.config.cache_directory = os.path.expanduser('cache_data')



classification_results = dict()

for task_id in classification_tasks:
    print(task_id)
    # get all the runs for this task
    try:
        runs = openml.runs.list_runs(task=[task_id])
        print("number runs: "+str(len(runs)))
        runs = openml.runs.get_runs(list(runs.keys()))
        runs_dict = [{"flow":run.flow_name, "score":run.evaluations['area_under_roc_curve'], "run":run} for run in runs]

        #sort by score
        runs_dict_sorted = sorted(runs_dict, key=lambda k: k['score'], reverse=True)

        # unique keep-first
        unique_ind = pa.DataFrame([run["run"].flow_id for run in runs_dict_sorted]).drop_duplicates(subset=None, keep='first', inplace=False).index.values
        runs_dict_unique = [runs_dict_sorted[i] for i in unique_ind]

        # keep top half
        if int(len(runs_dict_unique)) > 10:
            nr_keep = max(int(len(runs_dict_unique) / 4), 10)
            runs_dict_unique = runs_dict_unique[:nr_keep]

        classification_results[task_id] = runs_dict_unique
    except Exception as e: print(e)



with open('class_small_results.pickle', 'wb') as handle:
    pickle.dump(classification_results, handle, protocol=pickle.HIGHEST_PROTOCOL)



regression_results = dict()

for task_id in regression_tasks:
    print(task_id)
    # get all the runs for this task
    try:
        runs = openml.runs.list_runs(task=[task_id])
        print("number runs: "+str(len(runs)))
        runs = openml.runs.get_runs(list(runs.keys()))
        runs_dict = [{"flow":run.flow_name, "score":run.evaluations['mean_absolute_error'], "run":run} for run in runs]

        #sort by score
        runs_dict_sorted = sorted(runs_dict, key=lambda k: k['score'], reverse=False)

        # unique keep-first
        unique_ind = pa.DataFrame([run["run"].flow_id for run in runs_dict_sorted]).drop_duplicates(subset=None, keep='first', inplace=False).index.values
        runs_dict_unique = [runs_dict_sorted[i] for i in unique_ind]

        # keep top half
        if int(len(runs_dict_unique)) > 10:
            nr_keep = max(int(len(runs_dict_unique)/4),10)
            runs_dict_unique = runs_dict_unique[:nr_keep]
    except Exception as e: print(e)

    regression_results[task_id] = runs_dict_unique



with open('reg_small_results.pickle', 'wb') as handle:
    pickle.dump(regression_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

