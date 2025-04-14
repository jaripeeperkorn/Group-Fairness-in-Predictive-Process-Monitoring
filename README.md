# Fair OOPPM

## Description

This is the github repository containing the code, experiments and experimental results used in the paper:
> Achieving Group Fairness through Independence in Predictive Process Monitoring
> 
> Jari Peeperkorn and Simon De Vos

## Using the code

The folder DP_OOPPM contains the code to train and evaluate the predictive model. 

This also contains our pytorch compatible loss functions, and the metrics that can be used with any model's output.

Preprocessing contains all the function needed to preprocess the event logs.

```tutorials to be added```


## Rerunning the experiment

To recreate the results of the experiments (conda required), run:

```recreate_experiments.py```

## Rerunning the experiment part by part

The dependencies can be found in `environment.yaml`.

To recreate the folder structure needed, run following script:

```create_folder_structure.py```

The data used in the experiment can be found here (https://zenodo.org/records/8059489). It can be downloaded, by running the script:

```download_data.py```

To rerun the hyperparameter search run:

```hyperparameter_search_BCE.py```

To rerun the full experiment testing all sensitive parameters, run the code in:

```Experiment1_no_removal.py```

```Experiment1_with_removal.py```

To rerun the experiments testing the IPM based loss functions run:

```Experiment2.py```

```Experiment2_plot_curves.py```




