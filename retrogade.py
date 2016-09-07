"""
A script to extract synonyms from mammograms. 

"""
##############################################
# Author:   Soufyan Belkaid                  # 
#           VU University of Amsterdam       #
# Mail:     s.belkaid@student.vu.nl          #
# Version:  2.0                              #
##############################################

import difflib
import networkx as nx
import sys
import re
import os
# import matplotlib.pyplot as plt
from pattern.nl import singularize
import pygraphviz as pgv
import os
import yaml
import nltk
import logging
import time
import gensim
import pandas as pd
import fnmatch
import pickle
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from OpenDutchWordnet import Wn_grid_parser
from networkx.drawing.nx_agraph import graphviz_layout
from KafNafParserPy import KafNafParser
from networkx.readwrite import json_graph
from collections import Counter
from xml.etree.ElementTree import Element, SubElement, Comment
from xml.dom import minidom
from xml.etree import ElementTree
from lxml.etree import XMLSyntaxError
from subprocess import Popen, PIPE
from sklearn.cluster import KMeans
from difflib import get_close_matches
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
from collections import deque
from sklearn.feature_extraction.text import CountVectorizer
from difflib import get_close_matches


PARSED_FILES_DIR = 'parsed_files/'
VECTORIZER = CountVectorizer(ngram_range=(1,8), analyzer='word', lowercase=False)
ANALYZER = VECTORIZER.build_analyzer()
dutch_tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')
PATTERN = re.compile(r'[\./]')
PATH_TREE_TAGGER  ='~/Programming/terminology_extractor/libs/treetagger_plain2naf/treetagger_plain2naf.py'
CMD_EXTRACTOR_SCRIPT = '~/Programming/terminology_extractor/extract_patterns.py'
logging.basicConfig(filename='logging.log', level=logging.DEBUG)


def loadData(dir_name, pattern):
    nohyphen_files = []
    dir_names = []
    dir_paths = []
    for root, dirnames, filenames in os.walk(dir_name):
        dir_names.append(dirnames)
        dir_paths.append(root)
        for filename in fnmatch.filter(filenames, pattern):
            nohyphen_files.append(os.path.join(root, filename))
    return nohyphen_files, dir_names, dir_paths
                 
                 
def preprocess(files, tagged=False):
    """
    returns list of normalized sentences, sentence splitted and words have been lowered 
    :param list_files: A list of paths to the files to open and read
    :type list_files: list(str)
    :return: list of normalized words for every file
    :rtype: list(str)
    """
    normalized = []
    tagged_normalized = []
    for f in files:
        try:
            file_sentences = dutch_tokenizer.tokenize(open(f, 'r').read().decode('utf8').strip('\n').replace('.', ' '))
            # file_sentences = dutch_tokenizer.tokenize(open(f, 'r').read().decode('utf8').strip('\n'))
            # file_sentences = nltk.word_tokenize(open(f, 'r').read().decode('utf8').strip('\n'))
        except UnicodeDecodeError, e:
            print 'couldn\'t decode {}'.format(f)
            print e
            continue
        
        # tokenized_sentences = [regexp_tok.tokenize(sent) for sent in file_sentences]
        tokenized_sentences = [nltk.word_tokenize(doc) for doc in file_sentences]
        if tagged:
            tagged = [parser.find_tags(sent) for sent in tokenized_sentences]
            tagged_normalized.extend(tagged)
        container = []
        for sentence in tokenized_sentences:
            sentence_hack = []
            for word in sentence:
                sentence_hack.extend(re.split(r'[\./]', word.lower())) #stupid hack for erronous words such 
                #as cyste.classificatie reeel/compositie
            container.append(sentence_hack)
        normalized.extend(container)
        # normalized.extend([map(lambda x: x , [PATTERN.split(word.lower()) for word in sentence]) for sentence in tokenized_sentences])
    if tagged:   
        return normalized, tagged_normalized
    else:
        return normalized

    
def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def return_mods(words_found, path_to_db):
    """
    This functions should find the words and their modifier using 
    Ruben's terminology extractor. For now this function only works with 
    the first words found in WordNet by search_in_dwn

    :param words_found: list of words that are added to a xml pattern file.
    :type words_found: list
    :return container: a container of words and the output of the terminology etractor and 
    the word2vec model search of the words in words_found
    :rtype: dictionairy
    """
    top = Element('patterns')

    comment = Comment('Pattern file for terminology extractor')
    top.append(comment)
    #ALREADY SET-UP STORAGE FOR LATER USAGE 
    container = {}
    for word in words_found:
        container[word] = defaultdict(list) #INIT DEFAULTDICT TO STORE MODIFIERS
        child = SubElement(top, 'pattern', {'len':"2"})
        child.len = "2"
        ## ONLY SEARCHES FOR A N PATTERNS. IS THE REASON NOT ALL TERMS ARE FOUND AS ENTRY IN RETURNED DICT
        ## CAN ADD PATTERNS HERE
        SubElement(child, 'p',{
                "key":"pos",
                "position": "0",
                "values":"a"
            } )
        SubElement(child, 'p',{
                "key":"tokens",
                "position": "1",
                "values":word
            } )

    #STORE PATTERNS FILE
    if not os.path.isdir('patterns'):
        os.mkdir('patterns')

    logging.info("{} writing pattern file".format(time.strftime('%H:%M:%S')))
    file_name = os.path.abspath('.')+'/patterns/xml_pattern-{}.xml'.format(time.strftime('%d-%m-%y-%H:%M:%S'))
    with open(file_name, 'w', 0) as f: #0 is for not buffering
        f.write(prettify(top).encode('utf8'))
    
    ## CALL THE TERMINOLOGY EXTRACTOR WITH THE NEWLY CREATED PATTERNS
    cmd = ' '.join(['python', CMD_EXTRACTOR_SCRIPT, '-d', path_to_db, '-p', file_name])
    logging.info(cmd)
    logging.info("{} calling terminology extractor".format(time.strftime('%H:%M:%S')))
    process = Popen(cmd, stdout=PIPE, shell=True)
    output, err = process.communicate()    
    ##STORE ALL THE TERMS AND THEIR MODIFIERS IN A DICTIONAIRY
    for term_element in [line.split() for line in output.split('\n') if line]:
        freq, mod, term = term_element
        # print freq, term, word
        try:
            container[term]['modifiers'].append((mod,freq))
        except KeyError:
            print "not found in container: {}".format(term)
        
    for entry_term in container.keys():
        try:
            most_similar_words = model.most_similar(entry_term)
        except KeyError:
            print "not found in model: {}".format(entry_term)
            continue
        singularized = [singularize(w) for w in zip(*most_similar_words)[0]]
        container[entry_term]['similar'].extend(singularized)
        # container[entry_term]['similar'].extend(most_similar_words)
        # print normalize_word_input(most_similar_words)
    return container


def occurence(close_matches):
    """
    return the word with highest occurence according to trained Word2Vec model
    :param close_matches: list of words 
    :return: 

    """
    global model
    count_seq = []
    for word in close_matches:
        try:
            count_seq.append(model.vocab[word].count)
        except KeyError:
            count_seq.append(0)
    
    return close_matches[count_seq.index(max(count_seq))]


def normalize_word_input(word_list):
    """
    First remove words in the word_list that are alike. This is done by normalizing the words using 
    edit distance......
    How to find "correct" spelling? --> choose most occuring
    
    """
    #Words that haven't been tokenized properly, extra split on /
    splitted = set()
    for word_element in word_list:
        word = word_element[0]
        splitted.update(word.split('/'))
    
    most_freq = set()
    for word in splitted:
        close_matches = get_close_matches(word, splitted)
        ranked_occurence = occurence(close_matches)
        most_freq.add(ranked_occurence)
    
    ranked_on_similarity = []
    only_words = zip(*word_list)[0]
    for normalized in most_freq:
        try:
            ranked_on_similarity.append(word_list[only_words.index(normalized)])        
        except ValueError, e:
            pass
        
    return sorted(ranked_on_similarity, key=lambda x:x[1], reverse=True)


if __name__ == '__main__':
    files, folders, paths = loadData('sep_files/', '*')
    normalized_data = preprocess(files)
    model = gensim.models.Word2Vec(normalized_data) #model creation
    path_to_db = '~/Programming/terminology_extractor/ned_medDBCOPY.db' #added all the articles
    mods_and_synonyms = return_mods(['massa', 'kalk', 'tijd', 'locatie',
                         'plaats', 'asymmetrie', 'lateraal', 'bi-rads', 'scar',
                         'cyste', 'projectie'], path_to_db)

    df = pd.read_csv('documentation/voor Soufyan 2.csv')
    c_anatomie = df[df['anatomie']=='a']['Word [longest has 50 characters]']
    c_arch = df[df['archasym']=='a']['Word [longest has 50 characters]']
    c_locatie = df[df['plaats']=='p']['Word [longest has 50 characters]']
    c_tijd = df[df['tijd']=='t']['Word [longest has 50 characters]']
    c_massa = df[df['massa']=='m']['Word [longest has 50 characters]']
    attr_mass = df[df['bij massa']=='bm']['Word [longest has 50 characters]']
    c_path = df[df['path']=='pa']['Word [longest has 50 characters]']
    c_kalk = df[df['kalk']=='k']['Word [longest has 50 characters]']
    attr_kalk = df[df['bij calc']=='bc']['Word [longest has 50 characters]']

