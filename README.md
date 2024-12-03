# Fair OOPPM

## Description

This is the github repository containing the code, experiments and experimental results used in the paper:
> Achieving Group Fairness in Outcome Oriented Predictive Process Monitoring
> Jari Peeperkorn, Simon De Vos

## Using the code

The folder DP_OOPPM contains the code to train and evaluate the predictive model. 

This also contains our pytorch compatible loss functions, and the metrics that can be used with any model's output.

Preprocssing contains all the function needed to preprocess the event logs.

```tutorials to be added```


## Rerunning the experiment

The dependencies can be found in `environment.yaml`.

The data used in the experiment can be found here, and should be placed in a folder named Datasets: https://zenodo.org/records/8059489

To rerun the hyperparameter search run:

```hyperparameter_search_BCE```

To rerun the full experiment testing all sensitive parameters, run the code in:

```Experiment1_no_removal.py```

```Experiment1_with_removal.py```

To rerun the experiments testing the IPM based loss functions run:

```Experiment2.py```

```Experiment2_plot_curves.py```




