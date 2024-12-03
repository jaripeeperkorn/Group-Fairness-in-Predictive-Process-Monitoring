from sklearn.preprocessing import LabelEncoder

def integer_encode_categorical_data(df, cat_columns):
    """
    Encode categorical columns in a DataFrame as integers.

    This function takes a DataFrame and a list of categorical column names,
    and returns a new DataFrame where each specified column is transformed
    using integer encoding. A separate LabelEncoder is used for each column
    to convert categorical values into integer labels.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing categorical data.
        cat_columns (list of str): List of column names to be encoded.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns encoded as integers.
    """
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for column in cat_columns:
        le = LabelEncoder()  # Create a new LabelEncoder for each column
        df.loc[:, column] = le.fit_transform(df[column])  # Use .loc for assignment
    return df
