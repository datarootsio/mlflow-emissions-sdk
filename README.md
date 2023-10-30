[![maintained by dataroots](https://img.shields.io/badge/maintained%20by-dataroots-%2300b189)](https://dataroots.io)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/datarootsio/mlflow-emissions-sdk?label=version)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-360/)
![love](https://img.shields.io/badge/made_with-%E2%9D%A4%EF%B8%8F-blue)

![logo-dtr](https://user-images.githubusercontent.com/90327481/138892946-69b5f688-ff79-4b07-8864-44278b1695ca.png)

# mlflow_emissions_sdk

This package logs the carbon emissions of machine learning models.

## Overview

Training and running industrial machine learning models requires significant computing resources. The environment impacts are often overlooked if not neglected. To make ML engineers more aware about the carbon footprint of their projects we created this package.

The package seamlessly integrates the [CodeCarbon_package](https://codecarbon.io/) into the [MLFlow](https://mlflow.org/) MLOps package. Extending the logging capabilities of the latter with the carbon tracking methodology of the former.


## Installation

To get started, use the package manager [pip](https://pip.pypa.io/en/stable/) to install the mlflow-emissions-package:

`pip install mlflow-emissions-package`

Make sure that you have an mlflow client running:

`mlflow ui`

By default, the tracking uri is http://127.0.0.1:5000

To log your emissions you need to provide the tracker with the uri and the name of the experiment. If the experiment does not exist then the tracker will create a new experiment with the given name. Example

## Usage

```python
from pprint import pprint
import numpy as np
from sklearn.linear_model import LinearRegression
from mlflow_emissions_sdk.experiment_tracking_training import EmissionsTrackerMlflow


# Create a dictionary 
tracker_info = {
    "tracking_uri" : "http://127.0.0.1:5000",
    "experiment_name": "test_name",
    "run_name": "run_name",
    "flavor" : "sklearn"
}
# Prepare training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# Pick a model
model = LinearRegression()

# Instatiates the tracker
runner = EmissionsTrackerMlflow()
runner.read_params(tracker_info)

# Starts the emissions tracking
runner.start_training_job()

# Training the model
history = model.fit(X, y)

# Ends the tracking
runner.end_training_job()

```

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. You are free to use, modify, and distribute this software as per the terms of the license. If you find this project helpful, please consider giving it a star on GitHub.

## Acknowledgements
A special thanks to [Emmanuel](https://www.linkedin.com/in/emmanuelmenvouta/) for his supervision throughout this project. 

## Contact
- ali.al-gburi@dataroots.io
- ramiro@dataroots.io
- martijn@dataroots.io
- glenn.donghi@dataroots.io


