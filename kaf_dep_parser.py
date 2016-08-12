"""
This script parses KAF/NAF dependencies and creates a co-occurence matrix
(word * dependency-target). It also contains a function that saves the 
csr matrix in the dissect format http://clic.cimec.unitn.it/composes/toolkit/index.html
to be able to initiate a Space object. 
"""

##############################################
# Author:   Soufyan Belkaid                  # 
#           VU University of Amsterdam       #
# Mail:     s.belkaid@student.vu.nl          #
# Version:  1.0                              #
##############################################

import lxml.etree
import sys
import os
import re
import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy import io
from tempfile import TemporaryFile


usage = """
Usage: python kaf_dep_parser.py path_to_dir file_name_to_store

path_to_dir: path to directory with KAF/KAF files
file_name_to_store: file name that will be used to store matrix in dissect format
"""
DEP_RE = re.compile(r'(.*?)\((.*?)\)')


def construct_counter(dependency_list):
	"""
	construct a dictionairy where the key is the word and the values
	the counts of syntactic relations.
	:param dependency_list: list of dependency strings extracted from KAF/NAF
	:return: dict with the key being the row word and the values are the 
	dependencies it occurs with. 
	:rtype: dict
	"""
	dependency_dict = defaultdict(list)
	for doc in dependency_list:
		if doc:
			for entry in doc:
				try:
					target = entry[0][1]
					source = entry[1][1]
					dep = entry[1][0]
				except IndexError:
					continue
				
				dependency_dict[target].append((dep, source)) 

	for key in dependency_dict.keys():
		dependency_dict[key] = dict(Counter(dependency_dict[key]))
	return dependency_dict


def get_dependency(dependency_str):
	"""
	extract dependency from KAF/NAF 
	:param dependency_str: str containing Alpino dependency format
	:return: tuple containing the dependency and the target and context word
	"""
	if dependency_str.startswith(' - - / - -'):
		return None
	el = DEP_RE.findall(dependency_str)
	if not el:
		return None
	dep, obj = el[0]
	# print dep, obj
	 #max split is 1, begint links
	return zip(dep.strip(' ').split('/'), obj.split(',', 1))


def extractor(path_to_file):
    """
    :param: path to file
    :return: dependencies whitout None values
    :rtype: list
    """
    try:
        doc = lxml.etree.ElementTree(file=path_to_file)
    except lxml.etree.XMLSyntaxError, e:
    	print e
        return None

    doc_evaluator = lxml.etree.XPathEvaluator(doc)
    dependencies = doc_evaluator('//dep/comment()')
    return filter(None, [get_dependency(dep_str.text.encode('utf8')) 
    								for dep_str in dependencies])


def most_sim(word, matrix, mapping, top_n=5):
	"""
	:param matrix: dense X.todense()
	:param mapping: dicitionairy where the keys are the rows
	:return: sorted list of cosine distance values.
	:rtype: list
	"""

	word_list = mapping.keys()
	if not word in word_list:
		print "Word not in vocab"
		return None
	idx_target = word_list.index(word)
	vec_target = matrix[idx_target]
	cosines = []
	for idx_context, vec_context in enumerate(matrix):
		if idx_context == idx_target:
			continue
		# print idx_context, idx_target
		cos_val = cosine_similarity(vec_context, vec_target)
		if not cos_val:
			continue
		cosines.append((word_list[idx_context], cos_val))
	return sorted(cosines, key=lambda x:x[1], reverse=True)[:top_n]


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def save_dissect_format(file_name, csr_matrix, list_row_vals, dict_col_indx):
	"""
	Store csr matrix in mtx format to be processed by dissect module
	and create a semantic space.

	:param file_name: file name without extension
	:param csr_matrix: scipy sparse matrix
	:param list_row_vals: list of row entries in the matrix
	:param dict_col_indx: dictionairy containing the column entries and their
	indices. Returned by calling vectorizer.get_feature_names()
	"""
	outfile = TemporaryFile()
	io.mmwrite(outfile, csr_matrix)
	outfile.seek(0)
	f = outfile.read().split('\n')
	# print f
	splitted = [line.split(' ') for line in f]	
	s_mapping = {val: ' '.join(key).replace(' ', '_') for key,
				 val in dict_col_indx.items()} # _ added for MWE's
	#store dissect sm format
	with open(file_name+'.sm', 'w') as f:
		for el in splitted[3:-1]:
			try:
				f.write('%s %s %s\n' % (list_row_vals[int(el[0])],
					 s_mapping[int(el[1])], el[2]))
			except (KeyError, IndexError), e:
				print e, el
	#store dissect rows
	with open(file_name+'.rows', 'w') as f2:
		for row in list_row_vals:
			f2.write('%s\n' % row)
	#store dissect columns
	with open(file_name+'.cols', 'w') as f3:
		for col in s_mapping.values():
			f3.write('%s\n' % col)


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print usage
		sys.exit(1)

	file_dir = sys.argv[1]
	file_name = sys.argv[2]
	files = os.listdir(file_dir)
	extracted_deps = [extractor(os.path.join(file_dir,f)) for f in files]
	dependency_dict = construct_counter(extracted_deps)
	vectorizer = DictVectorizer()
	X = vectorizer.fit_transform(dependency_dict.values())
	mapping = vectorizer.vocabulary_
	 # _ are added for MWE's
	key_list = [key.replace(' ', '_') for key in dependency_dict.keys()]
	save_dissect_format(file_name, X, key_list, mapping)