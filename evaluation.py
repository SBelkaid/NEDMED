import sys
import os
import pandas as pd
from average_precision import *


usage = """
usage: python evaluation.py <path_to_file> <precision_at_k>

The file should be a csv file where for every pair of colums the first
colums are the predicted values and the second column are the correct values
"""

if len(sys.argv) < 3:
	print 'forgot a parameter'
	print usage
	sys.exit(1)

file_name = sys.argv[1]
k = int(sys.argv[2])
if not os.path.isfile(file_name):
	print usage
	sys.exit(1)

result_file = open(file_name, 'r')

df = pd.read_csv(result_file)

seq = df[df.columns.values[4:]]
data = [df[df[col].notnull()][col].values for col in df.columns]

real = []
predicted = []
idx_array = []
for idx, val in enumerate(data):
	if idx % 2 == 0:
		# print 'added to predicted: ', idx
		idx_array.append(idx)
		#doesn't work with numpy.array
		predicted.append([e.lower().strip(' ') for e in val.tolist()])
	else:
		real.append([e.lower().strip(' ') for e in val.tolist()])

map_val, map_val_array = mapk(real, predicted, k)
print '\nMAP val is: {} \n'.format(map_val)

for idx, val in zip(idx_array,map_val_array):
	print df.columns.values[idx], val



