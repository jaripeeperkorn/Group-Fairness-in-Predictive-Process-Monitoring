import Preprocessing.encode_categorical_data as encode_categorical_data
import Preprocessing.split_data as split_data
import Preprocessing.get_prefix_label_pairs as get_prefix_label_pairs

from sklearn.preprocessing import MinMaxScaler

import numpy as np

#! add settings other log types


# Define for the used event logs, which are categorical, which are numerical and change order etc.
def prepare_log(df, log_name, max_prefix_len, test_fraction=0.3, return_valdiation_set = False, validation_fraction=0.1, act_label = 'concept:name', case_id='case:concept:name', sensitive_column = 'case:gender', drop_sensitive=False):
    
    #added to be sure later on
    df = sort_log(df)
    
    if log_name == "hiring":
        #add outcome label
        df = add_label(df, "hiring", act_label, case_id)

        #REMOVE EVENT THAT COME AFTER THE OUTCOME EVENT
        df = df[df['valid_for_prefix'] == 1]

        #order features
        df, num_numerical_features, true_num_feature_list = clean_order_features(df, "hiring")
        #encode features
        df, vocsizes = encode_features(df, "hiring")


    elif log_name == "hospital":
        #add outcome label
        df = add_label(df, "hospital", act_label, case_id)

        #REMOVE EVENT THAT COME AFTER THE OUTCOME EVENT
        df = df[df['valid_for_prefix'] == 1]

        #order features
        df, num_numerical_features, true_num_feature_list  = clean_order_features(df, "hospital")
        #encode features
        df, vocsizes = encode_features(df, "hospital")


    elif log_name == "lending":
        #add outcome label
        df = add_label(df, "lending", act_label, case_id)

        #REMOVE EVENT THAT COME AFTER THE OUTCOME EVENT
        df = df[df['valid_for_prefix'] == 1]

        #order features
        df, num_numerical_features, true_num_feature_list  = clean_order_features(df, "lending")
        #encode features
        df, vocsizes = encode_features(df, "lending")

       
    elif log_name == "renting":
        #add outcome label
        df = add_label(df, "renting", act_label, case_id)

        #REMOVE EVENT THAT COME AFTER THE OUTCOME EVENT
        df = df[df['valid_for_prefix'] == 1]

        #order features
        df, num_numerical_features, true_num_feature_list  = clean_order_features(df, "renting")
        #encode features
        df, vocsizes = encode_features(df, "renting")

       
    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
           
    tr, te = split_data.train_test_split(df, test_fraction=test_fraction)

    # Print case statistics before prefix creation
    print(f"Training cases before prefix creation: {tr[case_id].nunique()}")
    print(f"Test cases before prefix creation: {te[case_id].nunique()}")


    #scale the true numerical data
    scaler = MinMaxScaler()
    tr_old_shape = tr[true_num_feature_list].shape
    tr[true_num_feature_list] = scaler.fit_transform(tr[true_num_feature_list].to_numpy().reshape(-1,1)).reshape(tr_old_shape)
    te_old_shape = te[true_num_feature_list].shape
    te[true_num_feature_list] = scaler.transform(te[true_num_feature_list].to_numpy().reshape(-1,1)).reshape(te_old_shape)


    #in this function the case_ID is droppend anyway
    trval_X, trval_y, trval_s, updated_max_prefix_length = get_prefix_label_pairs.create_pairs_train_sensitive(df=tr, max_prefix_length=max_prefix_len, sensitive_column=sensitive_column, drop_sensitive=drop_sensitive, case_id=case_id, outcome='outcome')
    te_X, te_y, te_s = get_prefix_label_pairs.create_pairs_test_sensitive(df=te, max_prefix_length=updated_max_prefix_length, sensitive_column=sensitive_column, drop_sensitive=drop_sensitive, case_id=case_id, outcome='outcome')

    # Print outcome statistics
    print("Training set outcome distribution:")
    print(f"  total number of prefixes in train: {len(trval_y)}")
    print(f"  outcome==0: {np.mean(np.array(trval_y) == 0) * 100:.2f}%")
    print(f"  outcome==1: {np.mean(np.array(trval_y) == 1) * 100:.2f}%")
    print("Test set outcome distribution:")
    print(f"  total number of prefixes in test: {len(te_y)}")
    print(f"  outcome==0: {np.mean(np.array(te_y) == 0) * 100:.2f}%")
    print(f"  outcome==1: {np.mean(np.array(te_y) == 1) * 100:.2f}%")

    if return_valdiation_set == False:
        return trval_X, trval_y, trval_s, te_X, te_y, te_s, vocsizes, num_numerical_features
    
    else:
        #split off validation set
        val_split_point = int(len(trval_y)*(1.0-validation_fraction))
        tr_X = trval_X[:val_split_point]
        tr_y = trval_y[:val_split_point]
        tr_s = trval_s[:val_split_point]
        val_X = trval_X[val_split_point:]
        val_y = trval_y[val_split_point:]
        val_s = trval_s[val_split_point:]
        return tr_X, tr_y, tr_s, val_X, val_y, val_s, te_X, te_y, te_s, vocsizes, num_numerical_features, updated_max_prefix_length


#!to do: add function that deletes cases from label definition onwards?

def add_label(df, log_name, act_label = 'concept:name', case_id='case:concept:name'):
    """
    Add outcome column to DataFrame based on a certain definition for different log types.

    Parameters:
    df (pd.DataFrame): Input DataFrame representing the event log.
    log_name (str): Type of log, e.g., "hiring", "hospital", "lending", or "renting".
    act_label (str): Column name for activity labels in the DataFrame (default is 'concept:name').
    case_id (str): Column name for case identifiers in the DataFrame (default is 'case:concept:name').

    Returns:
    pd.DataFrame: New DataFrame with an additional column indicating the presence of a specific activity based on the log type.
    """
    if log_name == "hiring":
        #we define the outcome based on whether make offer is present somewhere
        print(f'Preprocess {log_name} type event log, with outcome defined on presence of activity Make Job Offer.')
        new_df = add_label_activity_presence(df, act_label, case_id, 'Make Job Offer')

    elif log_name == "hospital":
        #! double check whether there are sets where both are present
        #we define the outcome based on whether treatment unsuccesful or treatment unsuccesful is initially reached
        print(f'Preprocess {log_name} type event log, with outcome defined on presence of treatment unsuccesful, even if succesful aftewards')
        new_df = add_label_activity_presence(df, act_label, case_id, 'Treatment unsuccesful')
    elif log_name == "lending":
        #we define the outcome based on whether the loan agreement is signed
        print(f'Preprocess {log_name} type event log, with outcome defined on presence of sign loan agreement')
        new_df = add_label_activity_presence(df, act_label, case_id, 'Sign Loan Agreement')
    elif log_name == "renting":
        #! we probably need to delete all events after sign contract
        #we define the outcome based on whether the contract is signed or proscpective tenant is rejected
        print(f'Preprocess {log_name} type event log, with outcome defined on presence of sign contract')
        new_df = add_label_activity_presence(df, act_label, case_id, 'Sign Contract')
    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    return new_df
    

def add_label_activity_presence(df, act_label, case_id, outcome_act):
    '''
    Add a new column to the DataFrame indicating whether each row's case ID matches the specified outcome action label,
    and include only activities up to but not including the first occurrence of the outcome action.
    
    Params:
    - df: DataFrame containing the data
    - act_label: Column name for the action label
    - case_id: Column name for the case ID
    - outcome_act: Value of the outcome action
    
    Returns:
    - DataFrame with additional 'outcome' and 'valid_for_prefix' columns
    '''
    # Identify case IDs where act_label is equal to outcome_act
    outcome_cases = df.loc[df[act_label] == outcome_act, case_id].unique()
    # Add a new 'outcome' column where the value is 1 if case_id is in outcome_cases, otherwise 0
    df['outcome'] = df[case_id].apply(lambda x: 1 if x in outcome_cases else 0)

    # Initialize 'valid_for_prefix' to mark activities before the outcome event in each case
    df['valid_for_prefix'] = 1  # Initialize all as valid initially

    # Iterate through each case and mark activities up to the first outcome event
    for case in outcome_cases:
        outcome_index = df[(df[case_id] == case) & (df[act_label] == outcome_act)].index
        if not outcome_index.empty:
            first_outcome_index = outcome_index[0]
            # Mark only activities before the first outcome event as valid
            df.loc[(df[case_id] == case) & (df.index >= first_outcome_index), 'valid_for_prefix'] = 0

    return df

def clean_order_features(df, log_name):
    #todo ORDER FEATURES, REMOVE DUPLICATES, CONVERT BOOLEAN TO NUMERICAL, MAKE SURE OUTCOME IS LAST
    if log_name == "hiring":
        #! we delete time for now
        categorical_features = ['concept:name', 'resource']
        binary_features = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected', 'case:religious']
        num_features = ['case:age', 'case:yearsOfEducation']
        num_numerical_features = len(binary_features) + len(num_features)
        for col in binary_features:
            #convert to integers instead of booleans
            df[col] = df[col].astype(int)
        #we add the case id in case we need it later again
        cols_order_filtered = categorical_features + binary_features + num_features + ['case:concept:name', 'outcome']
        df = df[cols_order_filtered]

    elif log_name == "hospital":
        #! we delete time for now
        categorical_features = ['concept:name', 'resource']
        binary_features = ['case:german speaking', 'case:private_insurance', 'case:underlying_condition', 'case:gender', 'case:citizen', 'protected']
        num_features = ['case:age']
        num_numerical_features = len(binary_features) + len(num_features)
        for col in binary_features:
            #convert to integers instead of booleans
            df[col] = df[col].astype(int)
        #we add the case id in case we need it later again
        cols_order_filtered = categorical_features + binary_features + num_features + ['case:concept:name', 'outcome']
        df = df[cols_order_filtered]
   

    elif log_name == "lending":
        #! we delete time for now
        categorical_features = ['concept:name', 'resource']
        binary_features = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected']
        num_features = ['case:age', 'case:yearsOfEducation', 'case:CreditScore']
        num_numerical_features = len(binary_features) + len(num_features)
        for col in binary_features:
            #convert to integers instead of booleans
            df[col] = df[col].astype(int)
        #we add the case id in case we need it later again
        cols_order_filtered = categorical_features + binary_features + num_features + ['case:concept:name', 'outcome']
        df = df[cols_order_filtered]

    elif log_name == "renting":
        #! we delete time for now
        categorical_features = ['concept:name', 'resource']
        binary_features = ['case:german speaking', 'case:gender', 'case:citizen', 'case:protected', 'case:married']
        num_features = ['case:age', 'case:yearsOfEducation']
        num_numerical_features = len(binary_features) + len(num_features)
        for col in binary_features:
            #convert to integers instead of booleans
            df[col] = df[col].astype(int)
        #we add the case id in case we need it later again
        cols_order_filtered = categorical_features + binary_features + num_features + ['case:concept:name', 'outcome']
        df = df[cols_order_filtered]

    
    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    
    #we also return list of true numericals to know which to minmaxscale
    return df, num_numerical_features, num_features 



def encode_features(df, log_name):
    voc_sizes = []
    if log_name == "hiring":
        cat_features = ['concept:name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    elif log_name == "hospital":
        # todo!
        #aanvullen?
        cat_features = ['concept:name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    elif log_name == "lending":
        # todo!
        #aanvullen?
        cat_features = ['concept:name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    elif log_name == "renting":
        # todo!
        #aanvullen?
        cat_features = ['concept:name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    
    return new_df, voc_sizes


def sort_log(df, case_id = 'case:concept:name', timestamp = 'time:timestamp'):
    """Sort events in event log such that cases that occur first are stored 
    first, and such that events within the same case are stored based on timestamp. 

    Parameters
    ----------
    df : _type_
        _description_
    case_id : str, optional
        _description_, by default 'case:concept:name'
    timestamp : str, optional
        _description_, by default 'time:timestamp'
    act_label : str, optional
        _description_, by default 'concept:name'
    """
    df_help = df.sort_values([case_id, timestamp], ascending = [True, True], kind='mergesort')
    # Now take first row of every case_id: this contains first stamp 
    df_first = df_help.drop_duplicates(subset = case_id)[[case_id, timestamp]]
    df_first = df_first.sort_values(timestamp, ascending = True, kind='mergesort')
    # Include integer index to sort on. 
    df_first['case_id_int'] = [i for i in range(len(df_first))]
    df_first = df_first.drop(timestamp, axis = 1)
    df = df.merge(df_first, on = case_id, how = 'left')
    df = df.sort_values(['case_id_int', timestamp], ascending = [True, True], kind='mergesort')
    df = df.drop('case_id_int', axis = 1)
    return df 