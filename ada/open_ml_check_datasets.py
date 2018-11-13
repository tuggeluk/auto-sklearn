"""
Scrape OpenMl for  top runs on each dataset
"""

import openml
import os
import pandas as pa
import numpy as np
import pickle
import traceback
openml.config.cache_directory = os.path.expanduser('cache_data')

classification_runs = list()
regression_runs  = list()

# check all datasets
datasets = openml.datasets.list_datasets(status="active")
tasks = openml.tasks.list_tasks()


# Nested Dicts storing the scraped metadata
# task_name/dataset_id/run_id
openml_metadata = dict()
openml_metadata["classification"] = dict()
openml_metadata["regression"] = dict()
openml_metadata["lc_prediction"] = dict()

nr_ds = 0


for dataset in datasets.keys():

    nr_ds +=1
    if nr_ds % 100 == 0:
        #storing data
        with open('openml_metadata'+str(nr_ds)+'.pickle', 'wb') as handle:
            pickle.dump(openml_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            openml_metadata = dict()
            openml_metadata["classification"] = dict()
            openml_metadata["regression"] = dict()
            openml_metadata["lc_prediction"] = dict()

    print("Getting Data for Dataset Nr: "+str(dataset))

    try:
        dataset_tasks = [tasks[k] for k in tasks if tasks[k]["did"]==dataset]
        if len(dataset_tasks) == 0:
            print("empty dataset")
            continue

        dataset_runs = [openml.runs.list_runs(task=[task["tid"]]) for task in dataset_tasks]
        run_lengths = [len(run) for run in dataset_runs]

        # max runs task
        task_id = run_lengths.index(max(run_lengths))
        max_runs = dataset_runs[task_id]
        if len(max_runs) == 0:
            print("empty dataset")
            continue

        runs_to_get = np.random.choice(list(max_runs.keys()), size=min(200,len(max_runs)), replace=False)


        # load runs
        runs_loaded = openml.runs.get_runs(runs_to_get)
        # get best runs for dataset
        print("runs for max. run task loaded")
        task_type = dataset_tasks[run_lengths.index(max(run_lengths))]["ttid"]
        print(dataset_tasks[run_lengths.index(max(run_lengths))]["task_type"])
        if task_type == 1: # classification
            type_key = "classification"
            runs_dict = [{"flow":run.flow_name, "score":run.evaluations['area_under_roc_curve'], "run":run} for run in runs_loaded]
        elif task_type == 2: # regression
            type_key = "regression"
            runs_dict = [{"flow":run.flow_name, "score":run.evaluations['mean_absolute_error'], "run":run} for run in runs_loaded]
        elif task_type == 3: # learning curve pred
            type_key = "lc_prediction"
            runs_dict = [{"flow":run.flow_name, "score":run.evaluations['predictive_accuracy'], "run":run} for run in runs_loaded]
        else:
            #not iterested in task
            continue


        #sort by score
        runs_dict_sorted = sorted(runs_dict, key=lambda k: k['score'], reverse=False)

        # unique keep-first
        unique_ind = pa.DataFrame([run["run"].flow_id for run in runs_dict_sorted]).drop_duplicates(subset=None, keep='first', inplace=False).index.values
        runs_dict_unique = [runs_dict_sorted[i] for i in unique_ind]

        # keep top half
        if int(len(runs_dict_unique)) > 10:
            nr_keep = max(int(len(runs_dict_unique) / 4), 10)
            runs_dict_unique = runs_dict_unique[:nr_keep]

        openml_metadata[type_key][dataset] = runs_dict_unique

    except Exception as e:
        with open('log.txt', 'a') as f:
            f.write(str(e))
            f.write(traceback.format_exc())


with open('openml_metadata'+str(nr_ds)+'.pickle', 'wb') as handle:
    pickle.dump(openml_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)