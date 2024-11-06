import pm4py
import pandas as pd


def import_xes(filename: str):
    """
    Reads an XES file using the pm4py library and returns the parsed event log.

    Parameters:
    filename (str): A string representing the path to the XES file to be read.

    Returns:
    pm4py.log.log.EventLog: Parsed event log object.
    """
    return pm4py.read_xes(filename)
