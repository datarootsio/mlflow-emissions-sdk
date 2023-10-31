import subprocess
import sys
import urllib.request
import time
import pytest
sys.path.append("./")
from mlflow_emissions_sdk.experiment_tracking_training import \
   EmissionsTrackerMlflow
from mlflow_emissions_sdk.utils import *

PROPER_EXP_DICT = { "tracking_uri": "http://127.0.0.1:5100",
                "experiment_name": "pytest_test",
                "run_name": "pytest_run",
                "flavor": "keras"
               }

BAD_EXP_DICT = { "tracking_uri": "http://127.0.0.1:5100",
                "experiment_nam": "pytest_test",
                "run_name": "pytest_run",
                "flavor": "keras"
               }


@pytest.fixture(scope='function')
def setup(request):
    try:
      response = urllib.request.urlopen('http://127.0.0.1:5100').getcode()

    except Exception:
       ss = subprocess.Popen(["mlflow ui -p 5100"], shell=True)
       subprocess.run(['sleep', '5'])

    def cleanup():
      pid = subprocess.run(["lsof -i :5100 | awk '{print $2}' | head -2 | grep -E '\d+'"], shell=True, capture_output=True)
      subprocess.run(['sleep', '1'])
      subprocess.run([f'kill {pid.stdout.decode().strip()}'], shell=True)
    request.addfinalizer(cleanup)


def test_tracking_uri_exists(setup):
   tracking_uri = 'http://127.0.0.1:5100'
   res = verify_tracker_uri_exists(tracking_uri)
   assert res == "Connected"

def test_tracking_uri_does_not_exist(setup):
   tracking_uri = 'http://127.0.0.1:5101'
   res = verify_tracker_uri_exists(tracking_uri)
   assert res == "Tracking uri is unreachable"

def test_reading_params_correct(setup):
   runner = EmissionsTrackerMlflow()
   runner.read_params(PROPER_EXP_DICT)


def test_reading_bad_params_incorrect(setup):
   runner = EmissionsTrackerMlflow()
   try:
      runner.read_params(BAD_EXP_DICT)
   except KeyError as e:
      assert e.args[0] == 'experiment_name'

def test_logging_emissions(setup):
   runner = EmissionsTrackerMlflow()
   runner.read_params(PROPER_EXP_DICT)
   runner.start_training_job()
   time.sleep(2)
   runner.end_training_job()
