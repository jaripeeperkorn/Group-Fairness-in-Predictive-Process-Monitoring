import Preprocessing.encode_categorical_data as encode_categorical_data
import Preprocessing.split_data as split_data
import Preprocessing.get_prefix_label_pairs as get_prefix_label_pairs

from sklearn.preprocessing import MinMaxScaler


#! add sensitive parameter extractor

#! add settings other log types



# Define for the used event logs, which are categorical, which are numerical and change order etc.
def prepare_log(df, log_name, max_prefix_len, test_fraction=0.3, act_label = 'concept:name', case_id='case:concept:name'):
    
    if log_name == "hiring":
        #add outcome label
        df = add_label(df, "hiring", act_label, case_id)
        #order features
        df, num_numerical_features, true_num_feature_list = clean_order_features(df, "hiring")
        #encode features
        df, vocsizes = encode_features(df, "hiring")


    elif log_name == "hospital":
        #add outcome label
        df = add_label(df, "hospital", act_label, case_id)
        #order features
        df, num_numerical_features, true_num_feature_list  = clean_order_features(df, "hospital")
        #encode features
        df, vocsizes = encode_features(df, "hospital")


    elif log_name == "lending":
        #add outcome label
        df = add_label(df, "lending", act_label, case_id)
        #order features
        df, num_numerical_features, true_num_feature_list  = clean_order_features(df, "lending")
        #encode features
        df, vocsizes = encode_features(df, "lending")

       
    elif log_name == "renting":
        #add outcome label
        df = add_label(df, "renting", act_label, case_id)
        #order features
        df, num_numerical_features, true_num_feature_list  = clean_order_features(df, "renting")
        #encode features
        df, vocsizes = encode_features(df, "renting")

       
    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    
    tr, te = split_data.train_test_split(df, test_fraction=test_fraction)


    #scale the true numerical data
    scaler = MinMaxScaler()
    tr_old_shape = tr[true_num_feature_list].shape
    tr[true_num_feature_list] = scaler.fit_transform(tr[true_num_feature_list].to_numpy().reshape(-1,1)).reshape(tr_old_shape)
    te_old_shape = te[true_num_feature_list].shape
    te[true_num_feature_list] = scaler.transform(te[true_num_feature_list].to_numpy().reshape(-1,1)).reshape(te_old_shape)

    # todo add possiblitiy validation set split here?

    #! TODO FIX FOLLOWING FUNCTION
    #in this function the case_ID is droppend anyway
    tr_X, tr_y = get_prefix_label_pairs.create_pairs(tr, max_prefix_len)
    te_X, te_y = get_prefix_label_pairs.create_pairs(te, max_prefix_len)

    #! ADD CONVERSION TO PYTORCH TENSOR?

    return tr_X, tr_y, te_X, te_y, vocsizes, num_numerical_features


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
        print(f'Preprocess {log_name} type event log, with outcome defined on presence of treatment succesful')
        new_df = add_label_activity_presence(df, act_label, case_id, 'Treatment succesful')
    elif log_name == "lending":
        #we define the outcome based on whether the loan agreement is signed
        print(f'Preprocess {log_name} type event log, with outcome defined on presence of sign loan agreement')
        new_df = add_label_activity_presence(df, act_label, case_id, 'Sign Loan Agreement')
    elif log_name == "renting":
        #! we probably need to delete all events after sign contract
        #we define the outcome based on whether the contract is signed or proscpective tenant is rejected
        print(f'Preprocess {log_name} type event log, with outcome defined on presence of sign loan agreement')
        new_df = add_label_activity_presence(df, act_label, case_id, 'Sign Contract')
    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    return new_df
    

def add_label_activity_presence(df, act_label, case_id, outcome_act):
    '''
    Add a new column to the DataFrame indicating whether each row's case ID matches the specified outcome action label.
    Params:
    - df: DataFrame containing the data
    - act_label: Column name for the action label
    - case_id: Column name for the case ID
    - outcome_act: Value of the outcome action
    Returns:
    - DataFrame with the additional 'outcome' column
    '''
    # Identify case IDs where act_label is equal to outcome_act
    outcome_cases = df.loc[df[act_label] == outcome_act, case_id].unique()
    # Add a new 'outcome' column where the value is 1 if case_id is in outcome_cases, otherwise 0
    df['outcome'] = df[case_id].apply(lambda x: 1 if x in outcome_cases else 0)
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
    
    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    
    #we also return list of true numericals to know which to minmaxscale
    return df, num_numerical_features, num_features 

'''
    elif log_name == "hospital":
        # todo!
   

    elif log_name == "lending":
        # todo!
        new_df = df

    elif log_name == "renting":
        # todo!
        new_df = df
    '''
    

def encode_features(df, log_name):
    voc_sizes = []
    if log_name == "hiring":
        # todo!
        #aanvullen?
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