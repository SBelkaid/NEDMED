"""
A script to extract synonyms from mammograms. 

"""
##############################################
# Author:   Soufyan Belkaid                  # 
#           VU University of Amsterdam       #
# Mail:     s.belkaid@student.vu.nl          #
# Version:  1.0                              #
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


def find(word):
    """
    return an overview of a word with all the similar words from a trained word2vec model and all
    the word with similar retrogade word order 
    """
    global mainGraph
    if not model.vocab.get(word):
        print "not in the vocabulary"
        return None
    print word
    print '\t', 'WORD2VEC SIMILARITY TO MAIN WORD', model.most_similar(word),'\n'
    print '\t\t', 'WORDS WITH SIMILAR RETROGADE WORD ORDER', mainGraph.edge[word].keys()
    for retro_sim in mainGraph.edge[word]:
        print '\t\t\t',retro_sim
        print '\t\t\t\t', 'MOST SIMILAR WORD2VEC', model.most_similar(retro_sim)
        print '\t\t\t\t\t', 'RETROGADE',mainGraph.edge.get(retro_sim).keys(),'\n'


def retrogade(word, graph):
    """
    return sorted list on retrogade word order, input is graph that is created from retrogade,
    the values of an entry are again sorted on word order
    """
    return [word[::-1] for word in sorted([word[::-1] \
        for alpha_word in graph.edge.get(word).keys() if word.isalpha()])\
             if len(alpha_word) > 3]
                 
                 
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


def return_word_collection(wordList):
    """
    This functions creates a word collection of the retrogade word order list. 
    The keys are the words and their values are lists with words they occur in. It's a way to see
    the morphological changes made to a word to represent different meaning. Also, it shows multi-
    words formed from the word: mamma --> mammareductieoperatie
    """
    word_collection=defaultdict(list)
    for word in wordList:
        if len(word)> 3:
            for other in wordList:
                if word in other and word != other:
                    if word:
                        word_collection[word].append(other)
            if not word_collection.get(word):
                word_collection[word]
    return word_collection


def singularize_words(word_list):
    singularized = []
    for word in word_list:
        singularized.append(singularize(word))
    return singularized


def change_format(collection):
    """
    changes format of a dictionairy to fit d3.js format visualization
    """
    struct = defaultdict(list)
    struct['name'] = 'ROOT'
    for node in collection.keys():
        children = collection[node]
        if children:
            changed_children = []
            for child in children:
                changed_children.append({'name':child, 'size':200})
            struct['children'].append({'name':node, 'children':changed_children})
        else:
            struct['children'].append({'name':node, 'size':200})
    return struct


def return_ordinate_synset(relations, rel='hyperonym'):
    global wordnet_synset2lemma
    ordinate_synset = dict()
    if relations:
        #list for the lemmas of the synset
        ordinate_synset[rel] = dict()
        #for every ordinate synset find lemmas
        for ordinate in relations:
            target_synset = dwn.synsets_find_synset(ordinate.get_target())
            id_target_synset = target_synset.get_id()
            lemmas_ordinate = wordnet_synset2lemma[id_target_synset]
            ordinate_synset[rel][id_target_synset]= defaultdict(list)
            ordinate_synset[rel][id_target_synset]['lemmas'].extend(lemmas_ordinate)
            ordinate_synset[rel][id_target_synset]['glosses'].extend(target_synset.get_glosses())
    return ordinate_synset

    
def return_relation(synset_id):
    """
    return a dictionairy containing the directly related hypernyms and hypoyms includiing their lemmas and 
    glosses. This functions calls return_ordinate_synset(relations) to find the lemmas and glosses of the 
    directly related synsets to the provided synset_id

    :param synset_id: A synset id
    :type synset_id: str
    :return: mapping containing ordinates
    :rtype: dictionairy
    """
    global dwn
    #Find ordinates of synset, returns Relation instances
    found_synset = dwn.synsets_find_synset(synset_id)
    if found_synset:
        hypernym_relations = found_synset.get_relations('has_hyperonym')
        hyponym_relations = found_synset.get_relations('has_hyponym')
        hyper, hypo = return_ordinate_synset(hypernym_relations, 'hypernym'),\
            return_ordinate_synset(hyponym_relations, 'hyponym')
    if all([hypo,hyper]):
        return dict(hypo, **hyper)
    
    
def get_synsets(word):
    """
    Given a word return a dictionairy containing second-level oordinates to the word.
    By finding all its direct hypernyms and then look for the hyper and hyponym of those synsets.
    It passes the synset_id of the directly related synset to the return_relation function
    """
    global dwn
    #Find all possibly related synsets by lemma 
    ss_list = []
    for lex_entry in dwn.lemma_get_generator(word):
        if not lex_entry:
            continue
        synset_id = lex_entry.get_synset_id()
        if not synset_id:
            continue
        ss_list.append({synset_id: return_relation(synset_id)})
    ordinate_synsets = {key: val for synset in ss_list for key, val in synset.items()}
    return {'word':word, 'possibly_related':ordinate_synsets}


def search_in_dwn(word_collection):
    """
    first singularize words and see if they can be found in wordnet. Return the ones 
    that were found and the ones that weren't for further processing. 
    """
    global dwn, wordnet_entries
    all_lemmas = wordnet_entries.keys()
    found_in_wn = set(word_collection).intersection(all_lemmas)
    difference = set(word_collection).difference(all_lemmas)
    singularized_and_found = set(singularize_words(difference)).intersection(all_lemmas)
    found_in_wn.update(singularized_and_found)
    return found_in_wn, set(word_collection).difference(found_in_wn)


def create_concept_tree(word_list, store=False):
    container = {}    
    for word in word_list:
        related_synsets = get_synsets(word)
        container[related_synsets['word']] = related_synsets['possibly_related']
        print "added subgraph"
    if store:
        pickle.dump(container, open('concep_treet.pickle', 'w+'))


def return_descending_frequency():
    """
    return list of words based on frequency in the texts
    """
    global model
    return sorted([(word, model.vocab.get(word).count) for word in model.vocab
        if word.isalpha()],
            key=lambda x:x[1], reverse=True)


def return_tagged(KNParser):
    """
    returns list of tuples containing token and pos-tag
    :param my_parser: xml string containing the NAF format from the treetagger called 
    by subproces. 
    :type my_parser: str
    :return: list of tuples containing the word and its pos tag
    :rtype: list(tuple)
    """
    tokens_terms = zip(KNParser.get_tokens(), KNParser.get_terms())
    sentences_and_pos =[]
    sentences_no_pos =[]
    sent_num = 1
    sent_with_pos = []
    sent_no_pos = []
    for token_el, term_el in tokens_terms:
        token = token_el.get_text().lower()
        token_sentence_number = token_el.get_sent()
        token_pos_tag = term_el.get_pos()
        combined = (token, token_pos_tag, token_sentence_number)

        if token_sentence_number > sent_num:
            sentences_and_pos.append(sent_with_pos)
            sentences_no_pos.append(sent_no_pos)
            sent_num = token_sentence_number
            sent_with_pos, sent_no_pos = [], []
        sent_with_pos.append(combined[:2])
        sent_no_pos.append(combined[0])
    
    return sentences_and_pos, sentences_no_pos


def treetagger(list_files):
    """
    This function returns a list of tuples containing the tokens and their corresponding pos tag.
    It calls the CLTL treetagger through subprocess.
    :param list_files: A list of paths to the files to open and read
    :type list_files: list(str)
    :return: a list of lists containing tuples with the word and its pos-tag.
    :rtype: list(list(tuple))
    """
    pos_tagged = []
    for f in list_files:
        file_name = f.split('/')[-1]
        input_file = open(f, 'r')
        cmd = ' '.join([PATH_TREE_TAGGER,'-l','nl'])
        logging.info("Preprocessing {}".format(file_name))
        p2 = Popen(cmd, stdin=input_file ,stdout=subprocess.PIPE, shell=True)
        
        ## TO STORE THE NAF TO FILES AS XML
        # output_file = PARSED_FILES_DIR+file_name+'_output.xml'
        # with open(output_file, 'w') as outs:
            # p2 = Popen(cmd, stdin=input_file, stdout=outs, shell=True)
            # output,err = p2.communicate()

        output,err = p2.communicate()
        treetagger_output = output.split("<?xml version='1.0' encoding='UTF-8'?>")[1]
        #THE KAFNAFPARSER doesn't return a parser if the input isn't a XML tree

 
def return_type(all_tagged_documents, pos='N'):
    """
    Loop through the parsed files and retrieve the token element given the PoS parameter. 

    :param all_tagged_documents: Tagged documents
    :type all_tagged_documents: List(list(tokens))
    :param pos: PoS to search for, default Noun
    :return: list of tuples containing the words and their PoS [(word, PoS)]
    :rtype: list(tuple)
    """
    word_type = []
    for doc in all_tagged_documents:
        for sentence in doc:
            for token_el in sentence:
                if token_el:
                    if token_el[1] == pos:
                        word_type.append(token_el)
    return zip(*word_type)[0]


def swc(seed_synset_id):
    """
    return the path to the top most node from a given synset.
    :param seed_synset_id: synset_id
    :type seedsynset_id: str
    :return: path to the top most node
    :rtype: deque(str)
    """
    pool = deque()
    pool.append(seed_synset_id)
    path = deque()
    G=nx.DiGraph()

    while pool:
        ss_id = pool.popleft()
        synset = dwn.synsets_find_synset(ss_id)
        hypo_rels = synset.get_relations('has_hyperonym')
        hypo_ids = [rel.get_target() for rel in hypo_rels]
        hypo_ids.append(ss_id)
        G.add_nodes_from(hypo_ids)
        [G.add_edge(hypernym, ss_id) for hypernym in hypo_ids]            
        print ss_id, 'has {} hypernyms'.format(len(hypo_rels))
        for hypo in hypo_rels:
            hypo_id = hypo.get_target()
            if hypo_id in path:
                print "already in path", hypo_id
                continue
            if hypo_id in pool:
                continue   
            pool.append(hypo_id)
        path.append(ss_id)
    G.remove_edges_from(G.selfloop_edges())
    return path, G


def return_synsets_given_word(word_list):
    """
    return the synsets of the words that were found in ODWN.
    :param word_list: a list of words that are in ODWN that can be iterated over
    :type word_list: a sequence that can be iterated over
    :return synsets_given_word: list of tuples containing (synset, original word, lemmas for the given ss)
    :rtype: list
    """
    global wordnet_entries
    global wordnet_synset2lemma
    synsets_given_word = []

    for word in word_list:
        synsets_for_word = list(wordnet_entries[word])

        if None in synsets_for_word:
            # print synsets_for_word
            synsets_for_word.remove(None)
        if not synsets_for_word: # stupid hack because for some the first element is None, thats why select last
            # print "continue"
            continue        
        ss = synsets_for_word[0] #take the first ss is easy
        lemmas = wordnet_synset2lemma[ss]
        synsets_given_word.append((ss, word ,lemmas))
    return synsets_given_word


def return_hyperonyms(ss_id):
    """
    still need to develop
    """
    global dwn_tops
    global dwn
    ss = dwn.synsets_find_synset(ss_id)
    hypos = ss.get_relations('has_hyperonym')
    if hypos:
        for hypo in hypos:
            print "hypernym", hypo.get_target()
    else:
        print "no hypernyms"
    if ss_id in dwn_tops:
        print "top node"
    else:
        print "not a top node"
    

def draw_graph_png(graph):
    hierarchy = pgv.AGraph(graph.edge, directed=True)
    hierarchy.layout(prog='dot')
    print "creating dot layout"
    hierarchy.draw('test.png')
    print "writing to dot file called: SomeDotFile.dot"
    hierarchy.write('SomeDotFile.dot')
    print "Opening SomeDotFile.dot"
    os.system('open test.png')


def show_all_paths(graph, shortest=False):
    """
    return all paths in a directed graph from source to target node
    """
    sorted_nodes = nx.topological_sort(graph)
    top_node = sorted_nodes[0]
    end_node = sorted_nodes[-1]
    all_paths = []
    for path in nx.all_simple_paths(graph, source=top_node, target=end_node):
        all_paths.append(path)
    if shortest:
        return min(all_paths)
    else:
        return all_paths


def return_most_occuring_lemma(ss):
    """
    Count the lemmas for the given set of ss and return the most occuring one.
    For lack of WSD
    :param ss: a set of ss
    :return: lemma with highest frequency
    :rtype: str
    """
    global wordnet_synset2lemma
    global wordnet_entries
    seed_lemmas = wordnet_synset2lemma[ss]
    # print lemmas
    synsets = set()
    for lemma in seed_lemmas:
        synsets.update(wordnet_entries[lemma])
    fruit_lemmas = []
    [fruit_lemmas.extend(wordnet_synset2lemma[synset]) for synset in synsets]
    lemma_counter = Counter(fruit_lemmas)
    if lemma_counter.values():
        return lemma_counter.most_common(1)[0][0]


def add_attributes(NXgraph, root_node = 'eng-30-00001740-n'):
    """
    Create a tree of the graph. A tree: |edges| + 1 == |nodes| . Then add attributes to the graph for display such as lemma and color.
    This show the distinction between the nodes found in the domain texts and the nodes added as hypernyms.
    """
    global synsets_found_DWN
    global tuples_original_ss_found
    global synsets_found_DWN
    global term_modifier_dict
    term_entries = term_modifier_dict.keys()
    tree = nx.dfs_tree(NXgraph, root_node)
    for node in tree:
        tree.node[node]['synset_id'] = node
        ## IF THE NODE IS FOUND IN THE TEXT 
        if node in synsets_found_DWN:
            tree.node[node]['color'] = 'blue'
            lemma = zip(*tuples_original_ss_found)[1][synsets_found_DWN.index(node)]
            tree.node[node]['lemma'] = lemma
            #CHECK IF MODIFIERS OF THE LEMMA CAN BE FOUND AND ADD THEM AS ATTRIBUTE
            if lemma in term_entries:
                tree.node[node]['modifiers'] = term_modifier_dict[lemma]     
        else:
            tree.node[node]['color'] = 'red'
            lemma_ = return_most_occuring_lemma(node)
            if not lemma_:
                tree.node[node]['lemma'] = node
            else:
                tree.node[node]['lemma'] = lemma_
    #id attribute is changed into name by tree_data
    nested_dict = json_graph.tree_data(tree, root=root_node, attrs={'children':'children', 'id':'name'})
    return tree, nested_dict 


def create_graph(all_synset_ids):
    """
    return a graph of the synsets ids given as parameter. 

    :param all_synset_ids: iterable of synsets
    :type seedsynset_id: list of strings
    :return: graph of nodes including their paths to the top most node 
    :rtype: networkx.DiGraph
    """
    global dwn
    pool = deque(all_synset_ids)
    path = deque()
    G=nx.DiGraph()
    logging.info('{} Initiated graph building'.format(time.strftime('%D %H:%M:%S')))
    while pool:
        ss_id = pool.popleft()
        synset = dwn.synsets_find_synset(ss_id)
        try:
            hypo_rels = synset.get_relations('has_hyperonym')
        except AttributeError:
            continue
        hypo_ids = [rel.get_target() for rel in hypo_rels]
        hypo_ids.append(ss_id)
        G.add_nodes_from(hypo_ids)
        [G.add_edge(hypernym, ss_id) for hypernym in hypo_ids]              
        # print ss_id, 'has {} hypernyms'.format(len(hypo_rels))
        for hypo in hypo_rels:
            hypo_id = hypo.get_target()
            if hypo_id in path:
                # print "already in path", hypo_id
                continue
            if hypo_id in pool:
                continue   
            pool.append(hypo_id)
        path.append(ss_id)
        logging.info('{} Added {} to graph'.format(time.strftime('%D %H:%M:%S'), ss_id))
    G.remove_edges_from(G.selfloop_edges())
    logging.info('{} Building graph'.format(time.strftime('%D %H:%M:%S')))
    return G


def derive_others(words):
    """
    This functions should take a list of words that were not found in WordNet in their surface form. The words
    should be reduced to the head of the word and the remaining as a modifier. The remaining modifier can be a head 
    of a compound word again. If that is the case then the word should be furter reduced into modifier and head. 
    """
    newly_found = []
    global model
    not_found_copy = list(words.copy())
    print len(not_found_copy)
    for word in not_found_copy:
        lemma = singularize(word)
        # similar_words = model.most_similar(word)
        if lemma != word:
            #continue further deducing with singularized word in place of old word
            not_found_copy[not_found_copy.index(word)] = lemma
            ss = wordnet_entries.get(lemma)
            if ss:
                #can be added to found_in_DWN
                #pop lemma instead of words, because it has been change to lemma
                word_popped = not_found_copy.pop(not_found_copy.index(lemma)) 
                word_type_word = return_most_occuring_WT(word)
                newly_found.append((word_popped, lemma, word_type_word, ss))
                

        # print '\n\n\n\n'
    print len(newly_found)
    print len(words)-len(not_found_copy)
    print len(not_found_copy)
    # compounds = find_head(not_found_copy)
    return newly_found, not_found_copy

def return_most_occuring_WT(word):
    """
    Given a word return the most occuring wordtype by the parser.
    This function uses a Counter to get the most occuring type. If counts are equal of a word is not found return N
    :param word: word found in the domain text
    :type word: str
    :return: most occuring word type
    :rtype: str
    """
    global noun_counter, verb_counter, adj_counter
    ct_occ = [noun_counter.get(word), verb_counter.get(word), adj_counter.get(word)]
    if not max(ct_occ):
        return 'N' #not found and not parsed by parser
    index = ct_occ.index(max(ct_occ))
    if index == 0:
        return 'N'
    if index == 1:
        return 'V'
    if index == 2:
        return 'A'


def find_head(compound_wordlist):
    """
    Given a list of compound words, for every compound return the head using wordnet lemmas deducting
    lemmas from the right moving to the beginning of the compound word. 
    """
    global wordnet_entries
    ss_ids = dwn.lemma2synsets
    head_compounds = defaultdict(list)
    max_length_head = {}
    for entry in wordnet_entries.keys():
        if not len(entry) > 2:
            continue
        for samenstelling in compound_wordlist:
            PoS = return_most_occuring_WT(samenstelling)
#            ss_entry = wordnet_entries[entry]
            if entry == samenstelling:
                continue
            if samenstelling.endswith(entry):
                
                remaining, head, end = samenstelling.partition(entry)
#                print samenstelling, entry, head
                synsets_given_entry = ss_ids[entry]
                #return the synsets with the same PoS
                bool_list = map(lambda x:x.endswith(PoS.lower()) if x else None, synsets_given_entry)
           
                for bool_val, ss in zip(bool_list, synsets_given_entry):
                    if bool_val:
                        print ss, samenstelling, head, PoS, remaining
                        head_compounds[samenstelling].append((head, ss, remaining))

#                if all(bool_list): # only correct PoS, hence the endswith
#                    head_compounds[samenstelling].append(head)
    for key, val in head_compounds.items():
            max_length_head[key]= max(val, key=lambda x: len(x[0]))

    # return head_compounds
    return max_length_head, head_compounds

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
            container[term]['modifiers'].append(mod)
        except KeyError:
            print "not found in container: {}".format(term)
        
    for entry_term in container.keys():
        try:
            most_similar_words = model.most_similar(entry_term)
        except KeyError:
            print "not found in model: {}".format(entry_term)
            continue
        container[entry_term]['similar'].extend(most_similar_words)
    return container


def create_sem_struct(mod_dict, w2v_model):
    """
    Output a graph with terms in the input dict as nodes and their modifiers as attributes.
    The input Word2Vec model functions as a way to find similarly used words in the text and 
    link them to each other. 
    """
    for term, mods in mod_dict.items():
        if term in w2v_model:
            similair_words = w2v_model.most_similar(term)
            PoS_term = return_most_occuring_WT(term)
            print PoS_term, term, similair_words[:4], '\n\n'


def print_cluster_output(amount_clusters):
    for cluster in xrange(amount_clusters):
        print '\nCluster %d' % cluster
        words=[]
        for i in xrange(0, len(word_centroid_map.values())):
            if (word_centroid_map.values()[i]==cluster):
                words.append(word_centroid_map.keys()[i])
        print words

    # all_clusters = defaultdict(list)
    # for key, val in word_centroid_map.items():
    #     all_clusters[val].append(key)

    # for cluster in all_clusters:
    #     if len(all_clusters[cluster]) >=4:
    #         print '\nCluster %d' % cluster
    #         print all_clusters[cluster]

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
        ranked_on_similarity.append(word_list[only_words.index(normalized)])        
    
    return sorted(ranked_on_similarity, key=lambda x:x[1], reverse=True)


if __name__ == '__main__':
    path_new_data = ''
    files, folders, paths = loadData('sep_files/', '*')
    normalized_data = preprocess(files)
    model = gensim.models.Word2Vec(normalized_data) #model creation
    extracted_terms = return_word_collection(model.vocab.keys()) #word list
    dwn = Wn_grid_parser(Wn_grid_parser.odwn) #ODWN instantiation
    dwn.load_synonyms_dicts() #load data
    dwn_tops = dwn.tops() #top nodes in WN
    
## #QUICK SEARCH DICTIONAIRY
    wordnet_entries = dwn.lemma2synsets
    wordnet_synset2lemma = dwn.synset2lemmas

    parsed_file_names = os.listdir(PARSED_FILES_DIR)
    KAFNAF_objects = [KafNafParser(PARSED_FILES_DIR+file_name) for file_name in parsed_file_names]
    tagged_sentences = [return_tagged(parser)[0] for parser in KAFNAF_objects]
    nouns = return_type(tagged_sentences, pos='N')
    verbs = return_type(tagged_sentences, pos='V')
    adjectives = return_type(tagged_sentences, pos='A')
    noun_counter, verb_counter, adj_counter = Counter(nouns), Counter(verbs), Counter(adjectives)

    
    found_in_wn, not_found = search_in_dwn(extracted_terms) #lemmas found in wordnet and not found
    # tuples_original_ss_found = return_synsets_given_word(found_in_wn) #tuples of synsets, words and lemmas
    # synsets_found_DWN = zip(*tuples_original_ss_found)[0] #Only synsets
#    red_node_list = set(zip(*tuples_original_ss_found)[0]).intersection(graph.node.keys()) #These are also in the radiology texts
#    blue_node_list = set(graph.node.keys()).difference(red_node_list) #These are the other nodes consisting of hypernyms. 
#    graph, graph_json = add_attributes(graph, red_node_list, blue_node_list)
    
    newly_found_nodes, still_not_found = derive_others(not_found)
    # ss_found_in_wn = synsets_found_DWN + zip(*return_synsets_given_word(zip(*newly_found_nodes)[0]))[0] #add the newly found nodes
    # graph = create_graph(ss_found_in_wn)
    # term_modifier_dict = return_mods(found_in_wn)
    # graph = yaml.load(open('vis/Graph_29-05-16-17:29:59.yaml', 'r'))
    # tree, nested_dict = add_attributes(graph)
    # d, e = find_head(still_not_found)
    # json.dump(nested_dict, open('graph_images/d3_vis/data/lemma_graph.json', 'w'))
    # graph = json.loads(open('graph_images/d3_vis/data/lemma_graph.json', 'r').read().decode('utf8'))
    # graph = yaml.load(open('Graph_23-05-16-15:28:13', 'r'))


## CALL CREATE_GRAPH TO BUILD A DIGRAPH THAT LOOKS FOR ALL THE HYPERNYMS OF THE SYNSETS FOUND BY LOOKING AT THE EXTRACTED WORDS FROM THE TEXT
    # graph = create_graph(synsets_found_DWN)
    # logging.info('{} Storing graph'.format(time.strftime('%D-%H:%M:%S')))
    # nx.write_yaml(graph, 'Graph_{}.yaml'.format(time.strftime('%d-%m-%y-%H:%M:%S')))
    # logging.info('{} Graph has been stored'.format(time.strftime('%D-%H:%M:%S')))


### NEED TO LOAD GRAPH BEFORE
    # import yaml
    # graph = yaml.load(open('Graph_29-05-16-17:29:59.yaml', 'r'))
    # red_node_list = set(zip(*tuples_original_ss_found)[0]).intersection(tree.node.keys())
    # blue_node_list = set(tree.node.keys()).difference(red_node_list)
    # pos = graphviz_layout(tree, prog='dot')
    # nx.draw_networkx_nodes(tree, pos, nodelist=blue_node_list, node_color='blue')
    # nx.draw_networkx_nodes(tree, pos, nodelist=red_node_list, node_color='red', alpha=0.6)
    # nx.draw_networkx_edges(tree, pos)

    # from networkx.readwrite import json_graph
    # d = json_graph.node_link_data(graph)
    # json.dump(d, open('GRAPH.json','w'))
##


    # parsed_file_names = os.listdir(PARSED_FILES_DIR)
    # KAFNAF_objects = [KafNafParser(PARSED_FILES_DIR+file_name) for file_name in parsed_file_names]
    # tagged_sentences = [return_tagged(parser)[0] for parser in KAFNAF_objects]
    # nouns = return_type(tagged_sentences, pos='N')
    # verbs = return_type(tagged_sentences, pos='V')
    # adjectives = return_type(tagged_sentences, pos='A')



## CLUSTERING
    ##VECTORIZING
    # #unnormalized
    # word_vectors = model.syn0
    # vectorizer = CountVectorizer(min_df=1)
    # matrix = vectorizer.fit_transform([' '.join(sent) for sent in normalized_data]).toarray()

    # # ##CLUSTERING
    # num_clusters= word_vectors.shape[0]/6
    # # # # # num_clusters= 50
    # kmeans_clustering = KMeans(n_clusters = 20)
    # # # t = time.strftime('%H:%M-%d-%m-%y')
    # idx = kmeans_clustering.fit_predict(matrix)
    # idx = kmeans_clustering.fit_predict(word_vectors)
    # word_centroid_map = dict(zip(model.index2word, idx))
    
    # # ##PRINTING
    # print_cluster_output(5)


## NEW STRATEGY
    path_to_db = '~/Programming/terminology_extractor/ned_medDB.db'
    path_to_db = '~/Programming/terminology_extractor/ned_medDBCOPY.db'
    mods_and_synonyms = return_mods(['massa', 'kalk', 'tijd', 'locatie',
                         'plaats', 'asymmetrie', 'lateraal', 'bi-rads'], path_to_db)

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

