"""
Basic usage of OpenMl
"""

import openml
import openmlkey
from sklearn import neighbors
import os
from scripts.update_metadata_util import  classification_tasks
import pandas as pa


for task_id in classification_tasks:
    print(task_id)
    # get all the runs for this task
    runs = openml.runs.list_runs(task=[task_id])
    runs = openml.runs.get_runs(list(runs.keys()))
    runs_dict = [{"flow":run.flow_name, "score":run.evaluations['area_under_roc_curve'], "run":run} for run in runs]

    #sort by score
    runs_dict_sorted = sorted(runs_dict, key=lambda k: k['score'], reverse=True)

    # unique keep-first
    unique_ind = pa.DataFrame([run["run"].flow_id for run in runs_dict_sorted]).drop_duplicates(subset=None, keep='first', inplace=False).index.values

    for run in runs:
        # add perfomance





# get runs for specific task

task = openml.tasks.get_task(31)
data = openml.datasets.get_dataset(task.dataset_id)
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
flow = openml.flows.sklearn_to_flow(clf)
run = openml.runs.run_flow_on_task(flow, task, avoid_duplicate_runs=False)
