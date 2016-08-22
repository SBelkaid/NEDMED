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


def store_dissect_format(file_name, csr_matrix, list_row, list_col):
	"""
	Store csr matrix in mtx format to be processed by dissect module
	and create a semantic space.

	:param file_name: file name without extension
	:param csr_matrix: scipy coordinate matrix
	:param list_row: list of row entries in the matrix
	:param list_col dictionairy containing the column entries and their
	indices. Returned by calling vectorizer.get_feature_names()
	"""
	col = csr_matrix.col#, len(csr_matrix.col)
	row = csr_matrix.row #, len(csr_matrix.row)
	data = csr_matrix.data #, len(csr_matrix.data)
	passed = []
	with open(file_name+'.sm', 'w') as f1:
		for i in range(len(data)):
			r,c,v = list_row[row[i]], list_col[col[i]], data[i]
			if not all([r,c,v]):
				passed.append(i)
				continue
			# print r,c,v
			try:
				f1.write('%s\t%s\t%s\n' % (list_row[row[i]], list_col[col[i]], data[i]))
			except (KeyError, IndexError), e:
				print e
	
	imp_order_cols = []
	with open(file_name+'.cols', 'w') as f2:	
		for i in range(len(col)):
			if not i in passed:
				if not list_col[col[i]] in imp_order_cols:
					imp_order_cols.append(list_col[col[i]])
					f2.write('%s\n' % list_col[col[i]])
	
	imp_order_rows = []
	with open(file_name+'.rows', 'w') as f3:
		for i in range(len(row)):
			if not i in passed:
				if not list_row[row[i]] in imp_order_rows:
					imp_order_rows.append(list_row[row[i]])
					f3.write('%s\n' % list_row[row[i]])


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
	reverse_mapping = {val: ' '.join(key).replace(' ', '_') for key,
			 val in mapping.items()} # _ added for MWE's
	store_dissect_format(file_name, X.tocoo(), key_list, reverse_mapping)