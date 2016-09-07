"""
This script converts the json format output in to a tabular format
necessary for the evaluation script
"""

##############################################
# Author:   Soufyan Belkaid                  # 
#           VU University of Amsterdam       #
# Mail:     s.belkaid@student.vu.nl          #
# Version:  1.0                              #
##############################################

import pandas as pd
import json
import numpy as np
import sys
from collections import defaultdict


usage = """
Usage: python write_to_csv.py path_to_result_file path_to_json

path_to_result_file: path to a file containing correct concepts 
path_to_json: json file containing the words used to find similar words to concepts
mods_and_syns output from retrogade.py or kaf_dep_parser.py
"""

MAPPING = defaultdict(str,
            {'asymmetrie': 'archasym',
             'bi-rads': 'bi-rads',
             'cyste': 'anatomie',
             'kalk': 'kalk',
             'lateraal': 'plaats',
             'massa': 'massa',
             'plaats': 'plaats',
             'projectie': 'proj',
             'scar': 'path',
             'tijd': 'tijd'})

path_res_file = sys.argv[1]
df = pd.read_json(open(sys.argv[2], 'r')) #mods_and_syns from retrogade or kaf_dep_parser

stacked = df.stack()
ts = stacked['similar']
result_format = pd.read_csv(open(path_res_file, 'r')).dropna(how='all')
data = pd.concat([pd.DataFrame(ts[row], columns=['{}'.format(row), 
		'{} similarity'.format(row)]) for row in ts.index], axis=1)
result_format = result_format[result_format.columns[result_format.columns.str.contains(
		'correct')]].dropna(how='all')

combined = []
only_terms = data[data.columns[~data.columns.str.contains('similarity')]]
for col in only_terms.columns:
	according_col = result_format.columns.str.contains(MAPPING[col])
	if np.any(according_col):
		combined.append(data[col])
		combined.append(result_format[result_format.columns[according_col]])

combined = pd.concat(combined, axis=1).dropna(how='all')



