import pandas as pd
from sklearn.preprocessing import StandardScaler

#! rewrite this: mayeb look at Brecht's code anyway

def create_and_preprocess_pairs_fixed_length(df, prefix_length, binary_features, categorical_features, numerical_features, 
                                             case_id='case:concept:name', outcome='outcome'):
    """
    Creates fixed-length prefixes for each case in the DataFrame, applies specified transformations, and splits into X and y data.
    
    Parameters:
    - df: DataFrame containing the event log data.
    - prefix_length: Integer indicating the fixed length of prefixes.
    - binary_features: List of column names for binary features to include in X_data.
    - categorical_features: List of column names for categorical features to include in X_data.
    - numerical_features: List of column names for numerical features to include in X_data.
    - case_id: Column name of the case ID in df (default 'case:concept:name').
    - outcome: Column name of the outcome in df (default 'outcome').

    Returns:
    - X_data: List of arrays containing selected and transformed features for each prefix.
    - y_data: List of outcome values corresponding to each prefix.
    """
    X_data = []
    y_data = []

    # Combine all specified feature types to create the filtered feature set
    selected_features = binary_features + categorical_features + numerical_features

    # Standardize numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Convert binary features to 0/1 (assuming boolean values True/False)
    df[binary_features] = df[binary_features].astype(int)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_features)

    # Update selected_features to reflect new one-hot encoded columns for categorical features
    selected_features = [col for col in df.columns if col in binary_features or col in numerical_features or 
                         col.startswith(tuple(categorical_features))]

    # Iterate over each unique case ID
    for case in df[case_id].unique():
        case_df = df[df[case_id] == case]

        # Filter the DataFrame to only include selected features after transformations
        case_df_filtered = case_df[selected_features]

        # Create prefixes for the given case
        for i in range(prefix_length - 1, len(case_df)):
            # Get the prefix slice with specified features only
            prefix_X = case_df_filtered.iloc[i - prefix_length:i].values
            # Append the prefix to X_data
            X_data.append(prefix_X)
            # Append the corresponding outcome value
            y_data.append(case_df.iloc[i - 1][outcome])
            
    return X_data, y_data

def create_pairs_fixed_length(df, prefix_length, binary_features, categorical_features, numerical_features, case_id='case:concept:name', outcome='outcome'):
    # Alternative function where we only use prefixes of a certain length, no padding
    # Split into X and y data with prefixes
    X_data = []
    y_data = []
    # Iterate over each unique case ID
    for case in df[case_id].unique():
        case_df = df[df[case_id] == case]
        selected_features = binary_features + categorical_features + numerical_features
        # Create prefixes for the given case
        for i in range(prefix_length-1, len(case_df)):
            prefix_X = case_df.iloc[i-prefix_length:i].values
            # Append the prefix to X_data
            X_data.append(prefix_X)
            # Append the corresponding outcome value
            y_data.append(case_df.iloc[i - 1][outcome])
    return X_data, y_data


def create_pairs(df, max_prefix_length, case_id='case:concept:name', outcome='outcome'):
    # Split into X and y data with prefixes
    X_data = []
    y_data = []
    # Iterate over each unique case ID
    for case in df[case_id].unique():
        case_df = df[df[case_id] == case]
        
        # Create prefixes for the given case
        for i in range(1, min(len(case_df), max_prefix_length) + 1):
            # Get prefix with last event the one at position i in case
            #! to do drop outcome columns etc.
            if i < max_prefix_length:
                prefix_X = case_df.iloc[:i].values
                #! to do add padding
            if i >= max_prefix_length:
                prefix_X = case_df.iloc[i-max_prefix_length:i].values
            # Append the prefix to X_data
            X_data.append(prefix_X)
            # Append the corresponding outcome value
            y_data.append(case_df.iloc[i - 1][outcome])
    return X_data, y_data