import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import Preprocessing.list_to_tensor as convert
import DP_OOPPM.train_model as train_model
import DP_OOPPM.evaluate_model as ev

import pandas as pd
import numpy as np

from itertools import product
import logging
import torch

import os

#code to find the sensitive parmeter for which the DP and other measures is the highest

def get_best_hyperparameter_combination(logname):
    """
    Load the hyperparameter tuning log and return the best hyperparameter combination based on AUC score.
    
    Parameters:
        logname (str): The base name of the log file (without the '_hyperparameter_tuning_results.csv' suffix).
        
    Returns:
        dict: The best hyperparameter combination and its corresponding AUC score.
    """
    # Define the log file path
    log_path = f"Hyperparameters/BCE/{logname}_hyperparameter_tuning_results.csv"
    
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


def run_sensitive_check(dataset_name, logname, max_prefix_length):
    if logname == 'hiring':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected', 'case:religious']
    elif logname == 'hospital':
        binarys = ['case:german speaking', 'case:private_insurance', 'case:underlying_condition', 'case:gender', 'case:citizen', 'protected']
    elif logname == 'lending':
        binarys = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected']
    log = imp.import_xes(dataset_name)

    tr_X, tr_y, tr_s, val_X, val_y, val_s, te_X, te_y, te_s, vocsizes, num_numerical_features = prepare.prepare_log(
        df=log, log_name=logname, max_prefix_len=8, test_fraction=0.2, 
        return_valdiation_set=True, validation_fraction=0.2,
        act_label='concept:name', case_id='case:concept:name', 
        sensitive_column='case:gender', drop_sensitive=False)
    
    # Convert data to tensors
    X_train, seq_len_train = convert.nested_list_to_tensor(tr_X)
    y_train = convert.list_to_tensor(tr_y).view(-1, 1)
    s_train = convert.list_to_tensor(tr_s)

    X_val, seq_len_val = convert.nested_list_to_tensor(val_X)
    y_val = convert.list_to_tensor(val_y).view(-1, 1)
    s_val = convert.list_to_tensor(val_s)

    hyperparams = get_best_hyperparameter_combination(logname)

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
        max_length=max_prefix_length, 
        learning_rate=hyperparams['learning_rate'], 
        max_epochs=300, 
        batch_size=hyperparams['batch_size'], 
        patience=30, 
        get_history=False, 
        X_val=X_val, 
        seq_len_val=seq_len_val, 
        y_val=y_val, 
        s_val=s_val
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize an empty list to store results for each sensitive attribute
    results_list = []

    for sensitive in binarys:
        _, _, _, _, _, _, te_X, te_y, te_s, _, _ = prepare.prepare_log(
        df=log, log_name=logname, max_prefix_len=8, test_fraction=0.2, 
        return_valdiation_set=True, validation_fraction=0.2,
        act_label='concept:name', case_id='case:concept:name', 
        sensitive_column=sensitive, drop_sensitive=False)

        X_te, seq_len_te = convert.nested_list_to_tensor(te_X)
        #y_te = convert.list_to_tensor(te_y).view(-1, 1)
        y_te = np.array(te_y)
        #s_te = convert.list_to_tensor(te_s)
        s_te = np.array(te_s)


        X_te, seq_len_te = X_te.to(device), seq_len_te.to(device)

        te_output = model(X_te, seq_len_te)
        te_np = te_output.detach().cpu().numpy()

        result = ev.get_evaluation(y_te, te_np, s_te)
        result['sensitive_attribute'] = sensitive  # Add the sensitive attribute name to the result

        # Append the result to the list
        results_list.append(result)

    # Convert the results list to a DataFrame and save to CSV
    results_df = pd.DataFrame(results_list)
    output_path = f"Sensitive_parameter_results/{logname}_sensitive_evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

run_sensitive_check('Datasets/hiring_log_high.xes', 'hiring', 8)

run_sensitive_check('Datasets/hospital_log_high.xes', 'hospital', 5)

run_sensitive_check('Datasets/lending_log_high.xes', 'lending', 5)