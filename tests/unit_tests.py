import sys
sys.path.append("./")
from mlflow_emissions_sdk.experiment_tracking_training import \
   EmissionsTrackerMlflow

from mlflow_emissions_sdk.utils import *

# import subprocess




FLAVORS = ["sklearn", "keras", "pytorch"]

def test_initializing_runner():
    runner = EmissionsTrackerMlflow()
    assert (
        str(type(runner))
        == "<class 'mlflow_emissions_sdk.experiment_tracking_training.EmissionsTrackerMlflow'>"
    )


def test_reading_tracking_params_correct():

    tracker_ui = {
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": "test_exp",
        "run_name": "test_run",
        "flavor": "keras",
    }
    verify_tracker_info(tracker_ui)

def test_reading_incorrect_tracking_params():
    tracker_ui = {
        "tracking_ur": "http://127.0.0.1:5000",
        "experiment_name": "test_exp",
        "run_name": "test_run",
        "flavor": "keras",
    }
    try:
        verify_tracker_info(tracker_ui)
    except KeyError as e:
        assert e.args[0] == 'tracking_uri'

def test_flavor_value_correct():
    flavor = "keras"
    verify_flavor_exists(flavor)

def test_flavor_value_fails():
    flavor = "jam"
    try:
        verify_flavor_exists(flavor)
    except ValueError as e:
        assert e.args(0) == f"Flavor can only be one of these values: {FLAVORS}"
