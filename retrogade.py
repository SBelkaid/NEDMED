"""
A script to extract synonyms from mammograms. Using Word2Vec

"""
##############################################
# Author:   Soufyan Belkaid                  # 
#           VU University of Amsterdam       #
# Mail:     s.belkaid@student.vu.nl          #
# Version:  2.0                              #
##############################################


import re
import os
import sys
import json
import nltk
import logging
import time
import gensim
import pandas as pd
import fnmatch
from xml.etree.ElementTree import Element, SubElement, Comment
from pattern.nl import singularize
from xml.dom import minidom
from xml.etree import ElementTree
from subprocess import Popen, PIPE
from difflib import get_close_matches
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


PARSED_FILES_DIR = 'parsed_files/'
VECTORIZER = CountVectorizer(ngram_range=(1,8), analyzer='word', lowercase=False)
ANALYZER = VECTORIZER.build_analyzer()
dutch_tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle')
PATTERN = re.compile(r'[\./]')
PATH_TREE_TAGGER  ='~/Programming/terminology_extractor/libs/treetagger_plain2naf/treetagger_plain2naf.py'
CMD_EXTRACTOR_SCRIPT = '~/Programming/terminology_extractor/extract_patterns.py'
logging.basicConfig(filename='logging.log', level=logging.DEBUG)

usage = """
python retrogade.py path_to_db path_to_files

path_to_db: path to the terminology database from terminology_extractor (index_files.py)
path_to_files: path to directory containing KAF/NAF files from convert_to_naf.sh
"""


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
            print>>sys.stderr, "not found in container: {}".format(term)
        
    for entry_term in container.keys():
        try:
            most_similar_words = model.most_similar(entry_term)
        except KeyError:
            print>>sys.stderr, "not found in model: {}".format(entry_term)
            continue
        singularized = [singularize(w) for w in zip(*most_similar_words)[0]]
        container[entry_term]['similar'].extend(singularized)
    return container


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print usage
        sys.exit(1)
    files, folders, paths = loadData(sys.argv[2], '*')
    normalized_data = preprocess(files)
    model = gensim.models.Word2Vec(normalized_data) #model creation
    # path_to_db = '~/Programming/terminology_extractor/ned_medDBCOPY.db' #added all the articles
    path_to_db = sys.argv[1]
    mods_and_synonyms = return_mods([u'massa', u'kalk', u'tijd', u'locatie',
                         u'plaats', u'asymmetrie', u'lateraal', u'bi-rads', u'scar',
                         u'cyste', u'projectie'], path_to_db)
    print>>sys.stdout, json.dumps(mods_and_synonyms)
