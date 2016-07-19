# NEDMED
=============

Description
----------
This is a research project containing a script that parses textual observation by radioligists and creates a Word2Vec model of the data. The objective of this project is to add structure to the way the observations are made. This is done by reducing the amount of words words a radiologist can use when referring to the same concept. The Word2Vec model is used to find words that 
occur in similar context and are regarded as possible synonyms. 
Apart from the Word2Vec model there's also a terminology extractor used in this project to extract word type patterns. 
The texts are parsed and n-grams (max length 8) are created. These word type patterns are then used to find the adjective of the possible synonym for a concept and stored as a modifier of that concept. 

Installation
-----------
Clone the repository from github

````shell
git clone git@github.com:SBelkaid/NEDMED.git
````
You will need to have installed the CLTL terminology extractor (link to Ruben's page). Follow the instructions on that page to install the terminology extractor. All the scripts are developed in Python, and use standard libraries. The only external
reference is to one wrapper for the TreeTagger pos-tagger and lemmatizer. You can install this by pip installing 
script.
```shell
git clone https://github.com/rubenIzquierdo/terminology_extractor.git
cd terminology_extractor
. install.sh
```

Best thing to do is install Anaconda https://www.continuum.io/downloads, contains more usefull modules, such as numpy, matplotlib and pandas. 

Usage
-----



Results
-------------


Testing
-------------
In some docstrings test have been made available. These can be run like so: 

```shell 
python -m doctest something.py -v
```

Visualistion
-------------


Future Work
------------


Contact
------

* Soufyan Belkaid
* s.belkaid@student.vu.nl
* Vrije University of Amsterdam

License
------
