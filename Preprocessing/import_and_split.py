import pandas as pd
import pm4py
import pm4py
import pandas as pd
import numpy as np

def import_xes(filename: str):
    """
    Reads an XES file using the pm4py library and returns the parsed event log.

    Parameters:
    filename (str): A string representing the path to the XES file to be read.

    Returns:
    pm4py.log.log.EventLog: Parsed event log object.
    """
    return pm4py.read_xes(filename)


# todo: Add strict temporal split if needed.
def train_test_split(df, test_fraction, case_id='case:concept:name', id_int=True):
    """
    Split the dataset into training and testing sets based on the specified test fraction.
    Since for now, we are working with artificial data this simple train testsplit suffices. 
    For real data a strict out-of-time split is recommended.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to split.
    test_fraction (float): Fraction of the dataset to be used as the test set.
    case_id (str): Column name for case IDs.
    id_int (bool): Flag to indicate whether to convert case IDs to integers.
    
    Returns:
    df_train (pd.DataFrame): Training set DataFrame.
    df_test (pd.DataFrame): Testing set DataFrame.
    """
    if id_int:
        df['case_id_int'] = df[case_id].astype(int)
        case_ids = sorted(df['case_id_int'].unique())
        split_point = int(len(case_ids) * (1.0 - test_fraction))
        df_train = df[df['case_id_int'].isin(case_ids[:split_point])].drop(columns=['case_id_int'])
        df_test = df[df['case_id_int'].isin(case_ids[split_point:])].drop(columns=['case_id_int'])
    else:
        case_ids = df[case_id].unique()
        split_point = int(len(case_ids) * (1.0 - test_fraction))
        df_train = df[df[case_id].isin(case_ids[:split_point])]
        df_test = df[df[case_id].isin(case_ids[split_point:])]

    return df_train, df_test
