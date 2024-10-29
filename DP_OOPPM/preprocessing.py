import pm4py

def get_train_test(filename, name_log, train_test_ratio, sensitive_parameter):
    log = import_xes(filename)
    traces_log = get_traces(log)
    outcomes = get_outcomes(traces_log, name_log)
    train_X_traces, train_y_traces, test_X_traces, text_y_traces = split_train_test(traces_log, outcomes, train_test_ratio)
    train_s_traces = get_sensitive_labels(train_X_traces)
    test_s_traces = get_sensitive_labels(test_X_traces)
    train_X, train_y, train_s = get_prefix_outcome_pairs(train_X_traces, train_y_traces, train_s_traces)
    test_X, test_y, test_s = get_prefix_outcome_pairs(test_X_traces, text_y_traces, test_s_traces)

    #todo: masking and padding + OH-encoding etc.

    return train_X, train_y, train_s, test_X, test_y, test_s



def import_xes(filename):
    log = pm4py.read_xes(filename)
    return log

def get_traces(log):
    #get traces from log
    print("getting traces")

def get_outcomes(log_list, name_log):
    outcome_list = []
    if name_log == "hiring":
        #we define the outcome based on whether make offer is present somewhere
        #! to do
        print(f'Preprocess {name_log} type event log, with outcome defined on presence of activity Make Job Offer.')
    elif name_log == "hospital":
        #! double check whether there are sets where both are present
        #we define the outcome based on whether treatment unsuccesful or treatment unsuccesful is initially reached
        #! to do
        print(f'Preprocess {name_log} type event log, with outcome defined on presence of either treatment succesful or treatment unsuccesful')
    elif name_log == "lending":
        #we define the outcome based on whether the loan agreement is signed
        #! to do
        print(f'Preprocess {name_log} type event log, with outcome defined on presence of sign loan agreement')
    elif name_log == "renting":
        #! we probably need to delete all events after sign contract
        #we define the outcome based on whether the contract is signed or proscpective tenant is rejected
        #! to do
        print(f'Preprocess {name_log} type event log, with outcome defined on presence of sign loan agreement')
    else:
        #todo: delete names of logs we won't do at the end
        raise ValueError("No valid event log type (of currently implemented) logs was given, try hiring, hospital, lending or renting.")
    return outcome_list

def get_sensitive_labels(traces, sensitive_parameter):
    #! to do
    sensitive_labels = []
    for trace in traces:
        #! this is not  how this works, fix this
        if trace[sensitive_parameter] == 'true':
            sensitive_labels.append(1)
        if trace[sensitive_parameter] == 'false':
            sensitive_labels.append(0)
        else:
            raise ValueError("The sensitive parameter value is not true or false as in the currently supported event logs.")

def split_train_test(traces, outcomes, ratio):
    split_point = int(len(traces)*ratio)
    return traces[0:split_point], outcomes[0:split_point], traces[split_point:-1], outcomes[split_point:-1]

def get_prefix_outcome_pairs(traces, outcomes, sensitives):
    prefixes = []
    outcomes_prefixes = []
    sensitives_prefixes = []
    for i in range(0, len(traces)):
        for j in range(0, len(trace[i])):
            prefixes.append([traces[i][0:j]])
            outcomes_prefixes.append[outcomes[i]]
            sensitives_prefixes.append[sensitives[i]]
    return prefixes, outcomes_prefixes, sensitives_prefixes