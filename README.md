# Fair OOPPM

## Description

This is the github repository containing the code, experiments and experimental results used in the paper "Independence Fairness in Outcome Oriented Predictive Process Monitoring".

## Using the code

The folder DP_OOPPM contains the code to train and evaluate the predictive model.
Preprocssing contains all the function needed to preprocess the event logs.

## Rerunning the experiment

The dependencies can be found in "environment.yaml".

The data used in the experiment can be found here: https://zenodo.org/records/8059489

To rerun the full experiment testing all sensitive parameters, run the code in:
"search_sensitive_parameters.py"
To rerun the experiments testing the IPM based loss functions run:
"experiment_loss_functions.py"

To rerun the hyperparameter search run:
"hyperparameter_search_BCE.py"


