from pprint import pprint

import numpy as np
from sklearn.linear_model import LinearRegression

from mlflow_emissions_sdk.experiment_tracking_training import EmissionsTrackerMlflow




tracker_info = {
    "tracking_uri" : "http://127.0.0.1:5000",
    "experiment_name": "test_name",
    "run_name": "test_run"
}
# prepare training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# train a model
model = LinearRegression()

# Instatiates the tracker
runner = EmissionsTrackerMlflow()
runner.read_params(tracker_info)

# Starts the emissions tracking
runner.start_training_job()

history = model.fit(X, y)

# Ends the tracking
runner.end_training_job()