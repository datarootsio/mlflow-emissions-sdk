"""
import urllib.request

FLAVORS = ["sklearn", "keras", "pytorch"]


def verify_tracker_info(tracker: dict) -> str:

    try:
        tracking_uri = tracker["tracking_uri"]
        experiment_name = tracker["experiment_name"]
        run_name = tracker["run_name"]
        flavor = tracker["flavor"]
    except KeyError:
        print(
            "Please input a dictionary with the keys 'tracking_uri', 'experiment_name', 'run_name', 'flavor'"
        )
        raise


def verify_tracker_uri_exists(tracker_uri: str):
    try:
        response_code = urllib.request.urlopen(tracker_uri).getcode()
        if response_code != 200:
            print("You do not have access to the tracking uri")
    except Exception:
        print("Tracking uri is unreachable")


def verify_flavor_exists(flavor):
    try:
        if flavor not in FLAVORS:
            raise ValueError(f"Flavor can only be one of these values: {FLAVORS}")
    except ValueError as e:
        print(e)


def verify_emission_tracker_is_instantiated(emissions_tracker):
    try:
        if emissions_tracker == None:
            raise Exception(
                "You need to start the EmissionsTrackerMlflow before starting a training job"
            )

    except Exception as e:
        print(e)


verify_flavor_exists("http://127.0.0.1:5000")
"""
