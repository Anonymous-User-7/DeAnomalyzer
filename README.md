# DeAnomalyzer

This repository contains the implementation of DeAnomalyzer that is used to reduce nondeterminism and inconsistency in anomaly detection algorithm implementations.

## Environments
* Python 3.9
* Matlab R2022b
* R 4.1.2
* Matlab Engine for Python `python -m pip install matlabengine`

## Usage
Add your dataset inside the `Dataset` folder. If the dataset contains Ground Truth, please rename that column to 'target'. The datasets used to test the tool can also be found in the Dataset folder. 

To run DeAnomalyzer on a dataset, run:

`python DeAnomalyzer.py file-name`