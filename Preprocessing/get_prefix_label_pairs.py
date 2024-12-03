def create_pairs_train(df, max_prefix_length, case_id='case:concept:name', outcome='outcome'):
    """
    Generate prefix-outcome pairs for training with a specified maximum prefix length.

    This function processes a DataFrame to create sequences of event prefixes and their
    corresponding outcomes, ensuring that the maximum prefix length does not exceed the
    length of the longest sequence. If the specified maximum prefix length is greater than
    the longest sequence, it is adjusted to match the longest sequence length.

    Parameters:
        df (DataFrame): The input DataFrame containing event data.
        max_prefix_length (int): The maximum length of prefixes to generate.
        case_id (str): The column name representing the case identifier. Default is 'case:concept:name'.
        outcome (str): The column name representing the outcome. Default is 'outcome'.

    Returns:
        tuple: A tuple containing:
            - prefixes (list): A list of event prefixes.
            - outcomes (list): A list of outcomes corresponding to each prefix.
            - max_prefix_length (int): The adjusted maximum prefix length.
    """
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
    """
    Generates prefix sequences, outcomes, and sensitive attribute values from the input DataFrame.

    This function processes the input DataFrame to create sequences of events (prefixes) up to a specified
    maximum prefix length. It also returns the corresponding outcome and sensitive attribute values for each
    prefix. If the specified maximum prefix length exceeds the length of the longest sequence, it is adjusted
    to match the longest sequence length.

    Parameters:
        df (DataFrame): The input DataFrame containing event logs.
        max_prefix_length (int): The maximum length of prefixes to generate.
        sensitive_column (str): The name of the column containing sensitive attribute values.
        drop_sensitive (bool, optional): Whether to drop the sensitive column from the sequences. Defaults to False.
        case_id (str, optional): The column name representing case identifiers. Defaults to 'case:concept:name'.
        outcome (str, optional): The column name representing outcome values. Defaults to 'outcome'.

    Returns:
        tuple: A tuple containing lists of prefixes, outcomes, sensitive attribute values, and the adjusted
        maximum prefix length.
    """   
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
    """
    Generates prefix sequences, outcomes, and sensitive attribute values from the input DataFrame.

    This function processes the input DataFrame to create sequences of events (prefixes) up to a specified
    maximum prefix length. It also returns the corresponding outcome and sensitive attribute values for each
    prefix. If the specified maximum prefix length exceeds the length of the longest sequence, it is adjusted
    to match the longest sequence length.

    Parameters:
        df (DataFrame): The input DataFrame containing event logs.
        max_prefix_length (int): The maximum length of prefixes to generate.
        sensitive_column (str): The name of the column containing sensitive attribute values.
        drop_sensitive (bool, optional): Whether to drop the sensitive column from the sequences. Defaults to False.
        case_id (str, optional): The column name representing case identifiers. Defaults to 'case:concept:name'.
        outcome (str, optional): The column name representing outcome values. Defaults to 'outcome'.

    Returns:
        tuple: A tuple containing lists of prefixes, outcomes, sensitive attribute values, and the adjusted
        maximum prefix length.
    """
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
    """
    Generate prefix-outcome-sensitive triples from a DataFrame for testing.

    This function creates sequences of prefixes, outcomes, and sensitive values
    from the input DataFrame, considering a maximum prefix length. It processes
    each case in the DataFrame, extracting prefixes up to the specified length
    and associating them with their respective outcomes and sensitive values.

    Parameters:
        df (DataFrame): The input DataFrame containing event logs.
        max_prefix_length (int): The maximum length of prefixes to generate.
        sensitive_column (str): The column name containing sensitive information.
        drop_sensitive (bool, optional): Whether to drop the sensitive column
            from the sequences. Defaults to False.
        case_id (str, optional): The column name identifying case IDs.
            Defaults to 'case:concept:name'.
        outcome (str, optional): The column name identifying outcomes.
            Defaults to 'outcome'.

    Returns:
        tuple: A tuple containing three lists:
            - prefixes (list): A list of prefix sequences.
            - outcomes (list): A list of outcome values corresponding to each prefix.
            - sensitives (list): A list of sensitive values corresponding to each prefix.
    """
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
    """
    Generate lists of sequences and outcomes from a DataFrame.

    This function processes a DataFrame to create two lists: one containing
    sequences of events for each case, and another containing the outcome
    of the last event in each sequence.

    Parameters:
        df (DataFrame): The input DataFrame containing event data.
        case_id (str): The column name representing the case identifier.
        outcome (str): The column name representing the outcome of each event.

    Returns:
        tuple: A tuple containing two lists:
            - X: A list of sequences, where each sequence is a list of events
            (excluding the case_id and outcome columns).
            - y: A list of outcomes, where each outcome corresponds to the last
            event in each sequence.
    """
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
    """
    Generate lists of sequences, outcomes, and sensitive values from a DataFrame.

    This function processes a DataFrame to create lists of sequences (X), 
    outcomes (y), and sensitive values (s) by grouping the data based on 
    the specified case identifier. Each sequence is derived from the events 
    within a case, optionally excluding a sensitive column. The outcome and 
    sensitive value are taken from the last event in each sequence.

    Parameters:
        df (DataFrame): The input DataFrame containing event data.
        sensitive_column (str): The name of the column containing sensitive data.
        drop_sensitive (bool): Whether to exclude the sensitive column from sequences.
        case_id (str): The column name used to identify different cases. Default is 'case:concept:name'.
        outcome (str): The column name representing the outcome of each case. Default is 'outcome'.

    Returns:
        tuple: A tuple containing three lists:
            - X: A list of sequences, where each sequence is a list of events.
            - y: A list of outcome values, one for each sequence.
            - s: A list of sensitive values, one for each sequence.
    """
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

