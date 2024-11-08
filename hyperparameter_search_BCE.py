import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import Preprocessing.list_to_tensor as convert
import DP_OOPPM.train_model as train_model

import pandas as pd
from sklearn.metrics import roc_auc_score
from itertools import product
import logging
import torch


# Main hyperparameter tuning function
def run_hyper(dataset_name, logname):
    # Log setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Data Preparation
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

    # Hyperparameter grids
    #num_layers_lst = [1, 2]
    #bidirectional_lst = [False, True]
    #LSTM_size_lst = [16, 32, 64]
    #batch_size_lst = [128] #we need a large size anyway later
    #learning_rate_lst = [0.0001, 0.001, 0.01]
    #dropout_lst = [0.0, 0.2, 0.4]

    # Hyperparameter grids
    num_layers_lst = [1, 2]
    bidirectional_lst = [False]
    LSTM_size_lst = [16]
    batch_size_lst = [128]
    learning_rate_lst = [0.0001]
    dropout_lst = [0.2]

    # Generate all combinations of hyperparameters
    hyperparameter_combinations = product(num_layers_lst, bidirectional_lst, LSTM_size_lst, batch_size_lst, 
                                           learning_rate_lst, dropout_lst)
    
    # List to store results
    results = []

    for combination in hyperparameter_combinations:
        num_layers, bidirectional, lstm_size, batch_size, learning_rate, dropout = combination
        
        logging.info(f"Training with hyperparameters: {combination}")

        # Initialize and train the model
        #we decreased the patience a bit, since e are just interested in best setup
        model = initialize_model(
            X_train=X_train, 
            seq_len_train=seq_len_train, 
            y_train=y_train, 
            s_train=s_train, 
            vocab_sizes=vocsizes, 
            num_numerical_features=num_numerical_features, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            lstm_size=lstm_size, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            dropout=dropout, 
            max_length=8, 
            max_epochs=300, 
            patience=20, 
            X_val=X_val, 
            seq_len_val=seq_len_val, 
            y_val=y_val, 
            s_val=s_val
        )
        
        # Evaluate the model
        auc = evaluate_model(model, X_val, y_val, seq_len_val)
        
        # Log results
        logging.info(f"AUC for {combination}: {auc}")
        
        # Save hyperparameters and AUC to results list
        results.append({
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'lstm_size': lstm_size,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'auc_score': auc
        })

        results_df = pd.DataFrame(results)
        # Save results to CSV
        results_df.to_csv(logname+'_hyperparameter_tuning_results.csv', index=False)
    
    # Convert results to a pandas DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(logname+'_hyperparameter_tuning_results.csv', index=False)
    
    # Return the results DataFrame (optional)
    return results_df

def initialize_model(X_train, seq_len_train, y_train, s_train, vocab_sizes, num_numerical_features, 
                     num_layers, bidirectional, lstm_size, batch_size, learning_rate, dropout, 
                     max_length, max_epochs, patience, X_val, seq_len_val, y_val, s_val):
    model = train_model.train_and_return_LSTM(
        X_train=X_train, 
        seq_len_train=seq_len_train, 
        y_train=y_train, 
        s_train=s_train, 
        loss_function='BCE', 
        vocab_sizes=vocab_sizes, 
        num_numerical_features=num_numerical_features, 
        dropout=dropout, 
        lstm_size=lstm_size, 
        num_lstm=num_layers, 
        bidirectional=bidirectional, 
        max_length=max_length, 
        learning_rate=learning_rate, 
        max_epochs=max_epochs, 
        batch_size=batch_size, 
        patience=patience, 
        get_history=False, 
        X_val=X_val, 
        seq_len_val=seq_len_val, 
        y_val=y_val, 
        s_val=s_val
    )
    return model

def evaluate_model(model, X_val, y_val, seq_len_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_val, seq_len_val = X_val.to(device), seq_len_val.to(device)

    val_output = model(X_val, seq_len_val)
    val_np = val_output.detach().cpu().numpy()

    # Get the ground truth and predictions
    y_gt = y_val.ravel()
    y_pred = val_np.ravel()

    # Compute AUC score
    auc = roc_auc_score(y_gt, y_pred)
    return auc


run_hyper('Datasets/hiring_log_high.xes', 'hiring')