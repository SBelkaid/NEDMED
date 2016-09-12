"""
This is a script to extract synonyms from mammograms using the DSM 
On the sintactic dependency between words. It reads a co-occurence matrix
and performs PPMI weighting, after which the Cosine is used as the similarity 
measure. 
"""

##############################################
# Author:   Soufyan Belkaid                  # 
#           VU University of Amsterdam       #
# Mail:     s.belkaid@student.vu.nl          #
# Version:  1.0                              #
##############################################



from composes.utils import io_utils
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.similarity.cos import CosSimilarity
from xml.etree.ElementTree import Element, SubElement, Comment
from collections import defaultdict
from xml.etree import ElementTree
from xml.dom import minidom
from subprocess import Popen, PIPE
import os
import time


usage = """
Usage: python dissect.py dissect_format_file_name

dissect_format_file_name: path to a file containing dissect format
"""

CMD_EXTRACTOR_SCRIPT = '~/Programming/terminology_extractor/extract_patterns.py'
file_name = sys.argv[1]

my_space = Space.build(data = file_name+".sm",
                       rows = file_name+".rows",
                       cols = file_name+".cols",
                       format = "sm")

my_space = my_space.apply(PpmiWeighting())
# print my_space.get_sim("spain", "netherlands", CosSimilarity())
# print my_space.get_neighbours('parenchymopbouw', 4, CosSimilarity())
# print my_space.get_neighbours('pension-n', 4, CosSimilarity())
# print my_space.id2row


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
    the PPMI model search of the words in words_found
    :rtype: dictionairy
    """
    global my_space
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

    # logging.info("{} writing pattern file".format(time.strftime('%H:%M:%S')))
    file_name = os.path.abspath('.')+'/patterns/xml_pattern-{}.xml'.format(time.strftime('%d-%m-%y-%H:%M:%S'))
    with open(file_name, 'w', 0) as f: #0 is for not buffering
        f.write(prettify(top).encode('utf8'))
    
    ## CALL THE TERMINOLOGY EXTRACTOR WITH THE NEWLY CREATED PATTERNS
    cmd = ' '.join(['python', CMD_EXTRACTOR_SCRIPT, '-d', path_to_db, '-p', file_name])
    # logging.info(cmd)
    # logging.info("{} calling terminology extractor".format(time.strftime('%H:%M:%S')))
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
            # most_similar_words = model.most_similar(entry_term)
            most_similar_words = my_space.get_neighbours(entry_term, 10, CosSimilarity())
        except KeyError:
            print "not found in model: {}".format(entry_term)
            continue
        container[entry_term]['similar'].extend(most_similar_words)
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
            print usage
            sys.exit(1)
	path_to_db = '~/Programming/terminology_extractor/ned_medDBCOPY.db' #added all the articles
	mods_and_synonyms = return_mods(['massa', 'kalk', 'tijd', 'locatie',
                         'plaats', 'asymmetrie', 'lateraal', 'bi-rads', 'scar',
                         'cyste', 'projectie'], path_to_db)
	# io_utils.save(my_space, "PPMI_.pkl")