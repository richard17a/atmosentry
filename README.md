[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
![Tests](https://github.com/richard17a/atmosentry/actions/workflows/python-package.yml/badge.svg)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-360/) 
[![Python 3.9](https://img.shields.io/badge/python-3.9-red.svg)](https://www.python.org/downloads/release/python-360/) 
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/downloads/release/python-360/)

# atmosentry

atmosentry is a numerical integrator that simulates the atmospheric entry of comets. For full description of the numerical model see [this article](https://). Key details are described below, focussing primarily on free model parameters.

## Setup

- Clone the Github repository
- Navigate to the project directory (/atmosentry)
- Install required packages
    - pip3 install -r requirements.txt
- Install the package
    - pip3 install .

## How to use atmosentry

The [examples/](./examples/) sub-directory contains several ipython notebooks that demonstrate the basic functionality of atmosentry.


## Model description

atmosentry includes specific semi-analytical prescriptions for the ablation, deformation, and fragmentation of comets as they travel through the atmosphere.

### mass ablation

mass loss due to ablation is described by the classical Bronshten 1983 par