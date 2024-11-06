from sklearn.preprocessing import LabelEncoder

def integer_encode_categorical_data(df, cat_columns):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    for column in cat_columns:
        le = LabelEncoder()  # Create a new LabelEncoder for each column
        df.loc[:, column] = le.fit_transform(df[column])  # Use .loc for assignment
    return df
