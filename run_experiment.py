import pm4py

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('Datasets/') if isfile(join('Datasets/', f))]

def extract_as_xes(filename):
    loc = 'Datasets/'+filename
    loc_new = loc[:-7] + ".xes"
    log = pm4py.read_xes(loc)
    pm4py.write.write_xes(log, loc_new)


for file in onlyfiles:
    print(file)
    extract_as_xes(file)