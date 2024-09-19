import pm4py

def import_xes(filename):
    log = pm4py.read_xes(filename)
    return log