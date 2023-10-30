from pathlib import Path

from setuptools import setup

# read the requirements
requirements_path = Path("requirements.txt")
# extract all the requirements to install
requirements = (
    requirements_path.read_text().split("\n") if requirements_path.exists() else ""
)
# run the installation of our mlflow_emissions_sdk
setup(
    name="mlflow_emissions_sdk",
    version="0.14",
    packages=["mlflow_emissions_sdk"],
    description="tracks carbon emissions and logs it to mlfow",
    install_requires=requirements,
    license="MIT",
)
