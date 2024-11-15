
#todo Add version that also return AND REMOVES sentive parameter lists?

def create_pairs_train(df, max_prefix_length, case_id='case:concept:name', outcome='outcome'):
    #Difference with non_train is that we check max length here and return it, so that it is always the same for text and validation as well.

    X, y = create_lists(df, case_id, outcome)

    #check if max_prefix_length > max length sequences
    max_seq_len = FindMaxLength(X)
    if max_prefix_length > max_seq_len:
        print("Maximum prefix length given is higher than longest case up until decision point.")
        print("Changed to that max length.")
        max_prefix_length = max_seq_len

    prefixes = []
    outcomes = []

    for i in range(len(X)):
        for j in range(1, len(X[i])):
            if j <= max_prefix_length:
                prefixes.append(X[i][0:j])
                outcomes.append(y[i])
            if j > max_prefix_length:
                prefixes.append(X[i][j-max_prefix_length:j])
                outcomes.append(y[i])

    return prefixes, outcomes, max_prefix_length

def create_pairs_train_sensitive(df, max_prefix_length, sensitive_column, drop_sensitive = False, case_id='case:concept:name', outcome='outcome'):
    #Difference with non_train is that we check max length here and return it, so that it is always the same for text and validation as well.

    X, y, s = create_lists_sensitive(df, sensitive_column, drop_sensitive, case_id, outcome)

    #check if max_prefix_length > max length sequences
    max_seq_len = FindMaxLength(X)
    if max_prefix_length > max_seq_len:
        print("Maximum prefix length given is higher than longest case up until decision point.")
        print("Changed to that max length.")
        max_prefix_length = max_seq_len

    prefixes = []
    outcomes = []
    sensitives = []

    for i in range(len(X)):
        for j in range(1, len(X[i])):
            if j <= max_prefix_length:
                prefixes.append(X[i][0:j])
                outcomes.append(y[i])
                sensitives.append(s[i])
            if j > max_prefix_length:
                prefixes.append(X[i][j-max_prefix_length:j])
                outcomes.append(y[i])
                sensitives.append(s[i])

    return prefixes, outcomes, sensitives, max_prefix_length





def create_pairs_test(df, max_prefix_length, case_id='case:concept:name', outcome='outcome'):
    #Difference with train is that we assume max length here

    X, y = create_lists(df, case_id, outcome)

    prefixes = []
    outcomes = []

    for i in range(len(X)):
        for j in range(1, len(X[i])):
            if j <= max_prefix_length:
                prefixes.append(X[i][0:j])
                outcomes.append(y[i])
            if j > max_prefix_length:
                prefixes.append(X[i][j-max_prefix_length:j])
                outcomes.append(y[i])

    return prefixes, outcomes


def create_pairs_test_sensitive(df, max_prefix_length, sensitive_column, drop_sensitive = False, case_id='case:concept:name', outcome='outcome'):
     #Difference with train is that we assume max length here

    X, y, s = create_lists_sensitive(df, sensitive_column, drop_sensitive, case_id, outcome)

    prefixes = []
    outcomes = []
    sensitives = []

    for i in range(len(X)):
        for j in range(1, len(X[i])):
            if j <= max_prefix_length:
                prefixes.append(X[i][0:j])
                outcomes.append(y[i])
                sensitives.append(s[i])
            if j > max_prefix_length:
                prefixes.append(X[i][j-max_prefix_length:j])
                outcomes.append(y[i])
                sensitives.append(s[i])

    return prefixes, outcomes, sensitives


def create_lists(df, case_id='case:concept:name', outcome='outcome'):
    X = []
    y = []
    # Group by the case_id column to separate sequences
    grouped = df.groupby(case_id)
    # Process each sequence
    for _, group in grouped:
        # Sort group by index to keep the original order of events within each case
        group = group.sort_index()
        
        # Drop the 'case:concept:name' and 'outcome' columns
        sequence = group.drop(columns=[case_id, outcome]).to_numpy().tolist()
        
        # Append the sequence to X
        X.append(sequence)
        
        # Append the outcome value of the last event in the sequence to y
        y.append(int(group[outcome].iloc[-1]))
        
    return X, y


def create_lists_sensitive(df, sensitive_column, drop_sensitive=False, case_id='case:concept:name', outcome='outcome'):
    X = []
    y = []
    s = []
    # Group by the case_id column to separate sequences
    grouped = df.groupby(case_id)
    # Process each sequence
    for _, group in grouped:
        # Sort group by index to keep the original order of events within each case
        group = group.sort_index()
        
        if drop_sensitive == False:
            # Drop the 'case:concept:name' and 'outcome' columns
            sequence = group.drop(columns=[case_id, outcome]).to_numpy().tolist()
        else:
            sequence = group.drop(columns=[case_id, outcome, sensitive_column]).to_numpy().tolist()
        
        # Append the sequence to X
        X.append(sequence)
        
        # Append the outcome value of the last event in the sequence to y
        y.append(int(group[outcome].iloc[-1]))

        #append sensitive
        s.append(int(group[sensitive_column].iloc[-1]))
        
    return X, y, s


def FindMaxLength(lst):
    maxList = max(lst, key = lambda i: len(i))
    maxLength = len(maxList)
     
    return maxLength

