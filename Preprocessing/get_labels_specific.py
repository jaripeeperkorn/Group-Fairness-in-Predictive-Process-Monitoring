#For the specific logs we use, there is no column 'outcome' yet, so we define it here.

import pandas as pd

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