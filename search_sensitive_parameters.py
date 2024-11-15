import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import Preprocessing.list_to_tensor as convert
import DP_OOPPM.train_model as train_model
import DP_OOPPM.evaluate_model as ev
import DP_OOPPM.plot_curves as plot_curves

import pandas as pd
import numpy as np


import torch

import os

#code to find the sensitive parmeter for which the DP and other measures is the highest

def get_best_hyperparameter_combination(logname, addendum):
    """
    Load the hyperparameter tuning log and return the best hyperparameter combination based on AUC score.
    
    Parameters:
        logname (str): The base name of the log file (without the '_hyperparameter_tuning_results.csv' suffix).
        
    Returns:
        dict: The best hyperparameter combination and its corresponding AUC score.
    """
    # Define the log file path
    log_path = f"Hyperparameters/BCE/{logname}_{addendum}_hyperparameter_tuning_results.csv"
    
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


def run_sensitive_check(dataset_name, logname, max_prefix_length, addendum):
    if logname == 'hiring':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected', 'case:religious']
    elif logname == 'hospital':
        binarys = ['case:german speaking', 'case:private_insurance', 'case:underlying_condition', 'case:gender', 'case:citizen', 'protected']
    elif logname == 'lending':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected']
    elif logname == 'renting':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected', 'case:married']
    log = imp.import_xes(dataset_name)

    X_train, seq_len_train, y_train, s_train, X_val, seq_len_val, y_val, s_val, X_te, seq_len_te, y_te, s_te, vocsizes, num_numerical_features, new_max_prefix_len = prepare.full_prep(filename=dataset_name, logname=logname, max_prefix_len=max_prefix_length, 
                                                                                                                                                                                       drop_sensitive=False, sensitive_column='case:gender')

    hyperparams = get_best_hyperparameter_combination(logname, addendum)

    model = train_model.train_and_return_LSTM(
        X_train=X_train, 
        seq_len_train=seq_len_train, 
        y_train=y_train, 
        s_train=s_train, 
        loss_function='BCE', 
        vocab_sizes=vocsizes, 
        num_numerical_features=num_numerical_features, 
        dropout=hyperparams['dropout'], 
        lstm_size=hyperparams['lstm_size'], 
        num_lstm=hyperparams['num_layers'], 
        bidirectional=hyperparams['bidirectional'], 
        max_length=new_max_prefix_len, 
        learning_rate=hyperparams['learning_rate'], 
        max_epochs=300, 
        batch_size=hyperparams['batch_size'], 
        patience=50, 
        get_history=False, 
        X_val=X_val, 
        seq_len_val=seq_len_val, 
        y_val=y_val, 
        s_val=s_val
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #prepare valdiations et needed for optimal threshold
    y_val_np = y_val.numpy()
    X_val, seq_len_val = X_val.to(device), seq_len_val.to(device)
    val_output = model(X_val, seq_len_val)
    val_output_np = val_output.detach().cpu().numpy()
    

    # Initialize an empty list to store results for each sensitive attribute
    results_list = []

    for sensitive in binarys:
        _, _, _, _, _, _, _, _, X_te, seq_len_te, y_te, s_te, _, _, _ = prepare.full_prep(filename=dataset_name, logname=logname, max_prefix_len=max_prefix_length, drop_sensitive=False, sensitive_column=sensitive)
        
        X_te, seq_len_te = X_te.to(device), seq_len_te.to(device)

        te_output = model(X_te, seq_len_te)
        te_np = te_output.detach().cpu().numpy()

        y_te = y_te.numpy()
        s_te = s_te.numpy()

        result = ev.get_evaluation_extented(y_te, te_np, s_te, y_val_np, val_output_np)

        result['sensitive_attribute'] = sensitive  # Add the sensitive attribute name to the result

        # Append the result to the list
        results_list.append(result)

        plot_filename = f"Sensitive_parameter_results/figs/{logname}_{addendum}_{sensitive}_plot.pdf"
        plot_filename = plot_filename.replace(" ", "").replace(":", "")

        plot_curves.plot_curves(te_np, s_te, sensitive, plot_filename)


    # Convert the results list to a DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    output_path = f"Sensitive_parameter_results/{logname}_{addendum}_sensitive_evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


run_sensitive_check('Datasets/hiring_log_high.xes', 'hiring', 6, 'high')

run_sensitive_check('Datasets/hiring_log_medium.xes', 'hiring', 6, 'medium')

run_sensitive_check('Datasets/hiring_log_low.xes', 'hiring', 6, 'low')

run_sensitive_check('Datasets/lending_log_high.xes', 'lending', 6, 'high')

run_sensitive_check('Datasets/lending_log_medium.xes', 'lending', 6,  'medium')

run_sensitive_check('Datasets/lending_log_low.xes', 'lending', 6,  'low')

run_sensitive_check('Datasets/renting_log_high.xes', 'renting', 6, 'high')

run_sensitive_check('Datasets/renting_log_medium.xes', 'renting', 6,  'medium')

run_sensitive_check('Datasets/renting_log_low.xes', 'renting', 6,  'low')
