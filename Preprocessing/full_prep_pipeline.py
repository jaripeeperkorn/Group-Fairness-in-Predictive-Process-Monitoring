import Preprocessing.import_data as imp
import Preprocessing.log_preparation_specific as prepare
import Preprocessing.list_to_tensor as convert

def full_prep(filename, logname, max_prefix_len, drop_sensitive, sensitive_column):
    """
    Executes the full preprocessing pipeline for an event log file.

    This function imports an XES file, prepares the event log for predictive modeling,
    and converts the data into PyTorch tensors suitable for LSTM input. It returns
    training, validation, and test datasets along with their sequence lengths, 
    vocabulary sizes, number of numerical features, and updated maximum prefix length.

    Parameters:
        filename (str): Path to the XES file to be imported.
        logname (str): Type of log for determining preprocessing steps.
        max_prefix_len (int): Maximum length of prefixes to generate.
        drop_sensitive (bool): Whether to drop the sensitive column.
        sensitive_column (str): Column name for sensitive attributes.

    Returns:
        tuple: Contains PyTorch tensors for training, validation, and test datasets,
        their sequence lengths, vocabulary sizes, number of numerical features, and
        updated maximum prefix length.
    """
    log = imp.import_xes(filename)
    tr_X, tr_y, tr_s, val_X, val_y, val_s, te_X, te_y, te_s, vocsizes, num_numerical_features, new_max_prefix_len = prepare.prepare_log(
        df=log, log_name=logname, max_prefix_len=max_prefix_len, test_fraction=0.2, 
        return_validation_set=True, validation_fraction=0.2,
        act_label='concept:name', case_id='case:concept:name', 
        sensitive_column=sensitive_column, drop_sensitive=drop_sensitive)
    X_train, seq_len_train = convert.nested_list_to_tensor(tr_X)
    y_train = convert.list_to_tensor(tr_y).view(-1, 1)
    s_train = convert.list_to_tensor(tr_s)

    X_val, seq_len_val = convert.nested_list_to_tensor(val_X)
    y_val = convert.list_to_tensor(val_y).view(-1, 1)
    s_val = convert.list_to_tensor(val_s)

    X_te, seq_len_te = convert.nested_list_to_tensor(te_X)
    y_te = convert.list_to_tensor(te_y).view(-1, 1)
    s_te = convert.list_to_tensor(te_s)

    return X_train, seq_len_train, y_train, s_train, X_val, seq_len_val, y_val, s_val, X_te, seq_len_te, y_te, s_te, vocsizes, num_numerical_features, new_max_prefix_len