import sys
import os
import pandas as pd
from average_precision import *


usage = """
usage: python evaluation.py <path_to_file>
"""

file_name = sys.argv[1]
if not os.path.isfile(file_name):
	print usage
	sys.exit(1)

result_file = open(file_name, 'r')

df = pd.read_csv(result_file)

seq = df[df.columns.values[4:]]
l = [df[df[col].notnull()][col].values for col in df.columns[4:]]

real = []
predicted = []
for idx, val in enumerate(l):
	if idx % 2 == 0:
		print 'added to predicted: ', idx
		#doesn't work with numpy.array
		predicted.append(val.tolist())
	else:
		real.append(val.tolist())

print 'MAP val is: {}'.format(mapk(real, predicted))

