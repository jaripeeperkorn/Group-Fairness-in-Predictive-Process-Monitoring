import encode_categorical_data
import split_data
import get_prefix_label_pairs


# Define for the used event logs, which are categorical, which are numerical and change order etc.
def prepare_log(df, log_name, max_prefix_len, test_fraction=0.3, act_label = 'concept:name', case_id='case:concept:name'):
    
    if log_name == "hiring":
        #add outcome label
        df = add_label(df, "hiring", act_label, case_id)
        #order features
        df = clean_features(df, "hiring")
        #encode features
        df, vocsizes = encode_features(df, "hiring")
        #! double check !!!!
        num_numerical_features = 6

    elif log_name == "hospital":
        #add outcome label
        df = add_label(df, "hospital", act_label, case_id)
        #order features
        df = clean_features(df, "hospital")
        #encode features
        df, vocsizes = encode_features(df, "hospital")
        #! double check !!!!
        num_numerical_features = 6

    elif log_name == "lending":
        #add outcome label
        df = add_label(df, "lending", act_label, case_id)
        #order features
        df = clean_features(df, "lending")
        #encode features
        df, vocsizes = encode_features(df, "lending")
        #! double check !!!!
        num_numerical_features = 6
       
    elif log_name == "renting":
        #add outcome label
        df = add_label(df, "renting", act_label, case_id)
        #order features
        df = clean_features(df, "renting")
        #encode features
        df, vocsizes = encode_features(df, "renting")
        #! double check !!!!
        num_numerical_features = 6
       
    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    
    tr, te = split_data.train_test_split(df, test_fraction=test_fraction)

    # todo add possiblitiy validation set split here?

    tr_X, tr_y = get_prefix_label_pairs.create_pairs(tr, max_prefix_len)
    te_X, te_y = get_prefix_label_pairs.create_pairs(te, max_prefix_len)

    #! ADD CONVERSION TO PYTORCH?

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

def clean_features(df, log_name):
    #todo ORDER FEATURES, REMOVE DUPLICATES, CONVERT BOOLEAN TO NUMERICAL, MAKE SURE OUTCOME IS LAST
    if log_name == "hiring":
        # todo!
        new_df = df

    elif log_name == "hospital":
        # todo!
        new_df = df

    elif log_name == "lending":
        # todo!
        new_df = df

    elif log_name == "renting":
        # todo!
        new_df = df

    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    return new_df


def encode_features(df, log_name):
    voc_sizes = []
    if log_name == "hiring":
        # todo!
        #aanvullen?
        cat_features = ['concept_name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    elif log_name == "hospital":
        # todo!
        #aanvullen?
        cat_features = ['concept_name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    elif log_name == "lending":
        # todo!
        #aanvullen?
        cat_features = ['concept_name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    elif log_name == "renting":
        # todo!
        #aanvullen?
        cat_features = ['concept_name', 'resource']
        for cat in cat_features:
            vocsize = df[cat].nunique()
            voc_sizes.append(vocsize)
        new_df = encode_categorical_data.integer_encode_categorical_data(df, cat_features)

    else:
        #todo: delete names of logs we didn't use
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    
    return new_df, voc_sizes