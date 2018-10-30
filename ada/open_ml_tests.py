"""
Basic usage of OpenMl
"""

import openml
from sklearn import neighbors
import os
task = openml.tasks.get_task(403)
data = openml.datasets.get_dataset(task.dataset_id)
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
flow = openml.flows.sklearn_to_flow(clf)
run = openml.runs.run_flow_on_task(flow, task, avoid_duplicate_runs=False)
