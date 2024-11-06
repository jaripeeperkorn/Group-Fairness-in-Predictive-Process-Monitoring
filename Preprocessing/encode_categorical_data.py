from sklearn.preprocessing import LabelEncoder

def integer_encode_categorical_data(df, cat_columns):
    le = LabelEncoder()
    for column in cat_columns:
        df[column] = le.fit_transform(df[column])
    return df
