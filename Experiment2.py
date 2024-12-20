import Preprocessing.full_prep_pipeline as prepare
import DP_OOPPM.train_model as train_model
import DP_OOPPM.evaluate_model as ev
import DP_OOPPM.plot_curves as plot_curves

import pandas as pd

import torch

def get_best_hyperparameter_combination(logname, addendum):
    """
    Load the hyperparameter tuning log and return the best hyperparameter combination based on AUC score.
    
    Parameters:
        logname (str): The base name of the log file (without the '_hyperparameter_tuning_results.csv' suffix).
        
    Returns:
        dict: The best hyperparameter combination and its corresponding AUC score.
    """
    # Define the log file path
    log_path = f"Results/Hyperparameters/BCE/{logname}_{addendum}_hyperparameter_tuning_results.csv"
    
    try:
        # Load the log data
        results_df = pd.read_csv(log_path)
        
        # Check if results are empty
        if results_df.empty:
            print("The results log is empty. No hyperparameter combinations have been recorded.")
            return None
        
        # Find the best hyperparameter combination by AUC score
        best_combination = results_df.loc[results_df['auc_score'].idxmax()]
        
        # Convert the best combination to a dictionary
        best_combination_dict = best_combination.to_dict()
        
        # Print and return the best hyperparameter combination
        print("Best Hyperparameter Combination:")
        print(best_combination_dict)
        
        return best_combination_dict
    
    except FileNotFoundError:
        print(f"Log file '{log_path}' not found. Please ensure the hyperparameter tuning has been run.")
        return None
    except KeyError as e:
        print(f"Expected column not found in log file: {e}")
        return None

def run_full_experiment(dataset_name, logname, addendum, max_prefix_length, sensitive_parameter, loss_fct):
    """
    Runs a full experiment pipeline including data preprocessing, model training, 
    evaluation, and result plotting.

    This function executes a series of steps to preprocess the dataset, train an 
    LSTM model with specified hyperparameters, evaluate the model's performance 
    using various metrics, and plot the probability distribution curves for a 
    sensitive feature. It iterates over different lambda values for specific loss 
    functions and saves the results to a CSV file.

    Parameters:
        dataset_name (str): The name of the dataset file.
        logname (str): The log name used for preprocessing and hyperparameter tuning.
        addendum (str): Additional identifier for the log file.
        max_prefix_length (int): Maximum prefix length for sequence generation.
        sensitive_parameter (str): Name of the sensitive attribute.
        loss_fct (str): The loss function to be used during model training.

    Returns:
        None
    """
    if loss_fct == "wasserstein":
        lambdas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    if loss_fct == "KL_divergence":
        lambdas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    X_train, seq_len_train, y_train, s_train, X_val, seq_len_val, y_val, s_val, X_te, seq_len_te, y_te, s_te, vocsizes, num_numerical_features, new_max_prefix_len = prepare.full_prep(filename=dataset_name, logname=logname, max_prefix_len=max_prefix_length, 
                                                                                                                                                                                       drop_sensitive=False, sensitive_column=sensitive_parameter)
    max_prefix_length = new_max_prefix_len

    hyperparams = get_best_hyperparameter_combination(logname, addendum)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #prepare valdiations set needed for optimal threshold
    y_val_np = y_val.numpy()
    X_val, seq_len_val = X_val.to(device), seq_len_val.to(device)

    #prepare test set further
    X_te, seq_len_te = X_te.to(device), seq_len_te.to(device)
    y_te = y_te.numpy()
    s_te = s_te.numpy()

    # Initialize an empty list to store results for lambda
    results_list = []

    for lam in lambdas:
        #! overrules hyperpameter search for batch size, needs to be larger for custom loss ftcs
        model = train_model.train_and_return_LSTM(X_train=X_train, 
                                                  seq_len_train=seq_len_train, 
                                                  y_train=y_train, 
                                                  s_train=s_train, 
                                                  loss_function=loss_fct, 
                                                  vocab_sizes=vocsizes, 
                                                  num_numerical_features=num_numerical_features, 
                                                  dropout=hyperparams['dropout'], 
                                                  lstm_size=hyperparams['lstm_size'], 
                                                  num_lstm=hyperparams['num_layers'], 
                                                  bidirectional=hyperparams['bidirectional'], 
                                                  max_length=max_prefix_length, 
                                                  learning_rate=hyperparams['learning_rate'], 
                                                  max_epochs=300, 
                                                  batch_size=512, 
                                                  patience=50, 
                                                  get_history=False, 
                                                  X_val=X_val, 
                                                  seq_len_val=seq_len_val, 
                                                  y_val=y_val, 
                                                  s_val=s_val,
                                                  balance_fair_BCE=lam)
        
        model.to(device)

        #prepare valdiations set needed for optimal threshold
        val_output = model(X_val, seq_len_val)
        val_output_np = val_output.detach().cpu().numpy()

        te_output = model(X_te, seq_len_te)
        te_np = te_output.detach().cpu().numpy()

        result = ev.get_evaluation_extented(y_te, te_np, s_te, y_val_np, val_output_np)

        result['lambda'] = lam  # Add the sensitive attribute name to the result

        # Append the result to the list
        results_list.append(result)

        plot_filename = f"Results/Experiment2_full_results/{loss_fct}/{logname}_{addendum}/{sensitive_parameter}/{str(lam)}_plot"
        plot_filename = plot_filename.replace(" ", "").replace(":", "").replace(".","")
        plot_filename = plot_filename + ".pdf"

        plot_curves.plot_curves(te_np, s_te, sensitive_parameter, plot_filename)

    # Convert the results list to a DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    output_path = f"Results/Experiment2_full_results/{loss_fct}/{logname}_{addendum}/{sensitive_parameter}/full_results"
    output_path = output_path.replace(" ", "").replace(":", "").replace(".","")
    output_path = output_path + ".csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
        


run_full_experiment(dataset_name='Datasets/lending_log_high.xes.gz', logname='lending', addendum='high', max_prefix_length=6, sensitive_parameter="case:protected", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/renting_log_high.xes.gz', logname='renting', addendum='high', max_prefix_length=6, sensitive_parameter="case:protected", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/hiring_log_high.xes.gz', logname='hiring', addendum='high', max_prefix_length=6, sensitive_parameter="case:protected", loss_fct='wasserstein')


run_full_experiment(dataset_name='Datasets/lending_log_high.xes.gz', logname='lending', addendum='high', max_prefix_length=6, sensitive_parameter="case:gender", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/renting_log_high.xes.gz', logname='renting', addendum='high', max_prefix_length=6, sensitive_parameter="case:gender", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/hiring_log_high.xes.gz', logname='hiring', addendum='high', max_prefix_length=6, sensitive_parameter="case:gender", loss_fct='wasserstein')


run_full_experiment(dataset_name='Datasets/hiring_log_high.xes.gz', logname='hiring', addendum='high', max_prefix_length=6, sensitive_parameter="case:religious", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/hiring_log_high.xes.gz', logname='hiring', addendum='high', max_prefix_length=6, sensitive_parameter="case:citizen", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/hiring_log_high.xes.gz', logname='hiring', addendum='high', max_prefix_length=6, sensitive_parameter="case:german speaking", loss_fct='wasserstein')


run_full_experiment(dataset_name='Datasets/lending_log_high.xes.gz', logname='lending', addendum='high', max_prefix_length=6, sensitive_parameter="case:citizen", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/lending_log_high.xes.gz', logname='lending', addendum='high', max_prefix_length=6, sensitive_parameter="case:german speaking", loss_fct='wasserstein')


run_full_experiment(dataset_name='Datasets/renting_log_high.xes.gz', logname='renting', addendum='high', max_prefix_length=6, sensitive_parameter="case:citizen", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/renting_log_high.xes.gz', logname='renting', addendum='high', max_prefix_length=6, sensitive_parameter="case:german speaking", loss_fct='wasserstein')

run_full_experiment(dataset_name='Datasets/renting_log_high.xes.gz', logname='renting', addendum='high', max_prefix_length=6, sensitive_parameter="case:married", loss_fct='wasserstein')

