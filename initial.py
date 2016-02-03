import pandas as pd
import tempfile
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nltk.corpus import stopwords
from sklearn.preprocessing import label_binarize
import os


plt.style.use('ggplot')
f = open('gestructureerde verslaglegging.xlsx', 'r')
STOPWORDS_DUTCH = stopwords.words('dutch')

def createFoldersWithFiles(verslagen):
	for annotation in verslagen.keys():
		statement = 'mkdir /sep_files/%s' % annotation
		print statement
		os.system(dir_name)
		for text in verslagen[annotation]:
			file_name = '\t/sep_files/%s/%s' % (annotation, tempfile.NamedTemporaryFile().name.split('/')[-1])
			print file_name
			with open('file_name', 'w') as f:
				f.write(text)

def reorderData(all_data, labels, sample=400):
	Y_train = []
	X_train = []
	Y_test = []
	X_test = []
	for label in labels:
		enkel_label_texten = all_data[label]

		if len(enkel_label_texten) < sample:
			X_train.extend(enkel_label_texten)
			for i in range(len(enkel_label_texten)):
				Y_train.append([label])
		else:
			X_train.extend(enkel_label_texten[:sample])
			remaining = len(enkel_label_texten[sample:])
			for i in range(sample):Y_train.append([label])
			if remaining < sample/2:
				X_test.extend(enkel_label_texten[sample:sample+remaining])
				print 'extending X_test with {}'.format(len(enkel_label_texten[sample:sample+remaining]))
				for i in range(remaining):Y_test.append([label])
				print 'extending Y_test with {}'.format(remaining)
			if remaining > sample/2:
				X_test.extend(enkel_label_texten[sample:sample+(sample/2)])
				print 'extending X_test with {}'.format(len(enkel_label_texten[sample:sample+(sample/2)]))
				for i in range(sample/2):Y_test.append([label])
				print 'extending Y_test with 50'

	return X_train, Y_train, X_test, Y_test

def trainAndPredict(X_train, Y_train, X_test, Y_test):
	lb = preprocessing.MultiLabelBinarizer(classes=enkele_labels)
	Y = lb.fit_transform(Y_train)
	Y_test = lb.fit_transform(Y_test)

	classifier = Pipeline([
	('vectorizer', CountVectorizer(stop_words=STOPWORDS_DUTCH)),
	('tfidf', TfidfTransformer()),
	('clf', OneVsRestClassifier(LinearSVC()))
	])

	classifier.fit(X_train, Y)
	predicted = classifier.predict(X_test)
	# print predicted
	binarized_Y_test = label_binarize(Y_test, enkele_labels)
	print "Accuracy Score: ",accuracy_score(Y_test, predicted)
	# print "Precision Score: ",precision_score(binarized_Y_test, predicted, None) data still has to be transformed
	# print "Recall Score: ",recall_score(binarized_Y_test, predicted, None) 
	return classifier, lb, predicted

def visFreqAnnotations():
	'''bar vis of the 10 most occuring code annotations'''
	N = 10
	ind = range(N)
	width = 0.35
	plt.bar(ind, ordered_annotations[:N])
	plt.xticks(ind, ordered_annotations[:N].index, rotation=17)

def showFreq(corpus, clf, top=10):
	'''
	Most occuring words
	'''
	vect = clf.named_steps.get('vectorizer')
	X = vect.fit_transform(corpus) #vectorize words
	vect = clf.named_steps.get('vectorizer')
	zip(vect.get_feature_names(),np.asarray(X.sum(axis=0)).ravel())
	return sorted(zip(vect.get_feature_names(),np.asarray(X.sum(axis=0)).ravel()),\
			 key=lambda x: x[1], reverse=True)[:top]

def categoriesFiles(code_anotation_list):
	for x in code_anotation_list:
		os.system('mkdir sep_files/'+x+'/')

		for e in df[df['CodeAnnotaties']== x]['EindverslagHuisartsTekst'].values:
			# rand_file_name = 'sep_files/'+x+'/'+tempfile.NamedTemporaryFile().name.split('/')[-1]
			rand_file_name = 'sep_files/'+x+'/'+x
			with open(rand_file_name, 'a') as f:
				f.write('\n'+e.encode('utf8'))

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("\n%s: %s" % (enkele_labels[class_label],
              " ".join(feature_names[j] for j in top10)))


df = pd.read_excel(f).dropna()
#freq different possible code combinations
ordered_annotations = df['CodeAnnotaties'].value_counts() 
#freq different possible described combinations
description_annotations = df['OmsAnnotaties'].value_counts()
#amount of unique code_annotations
unique_annotations = len(set(ordered_annotations)) 
#beschrijvingen die bij annotatie massa horen
decription_massa = df[df['OmsAnnotaties']=='Massa']

verzameling_categorie = defaultdict(list)

for annotatie, text in df[['CodeAnnotaties', 'EindverslagHuisartsTekst']].values:
    verzameling_categorie[annotatie].append(text)

enkele_labels = list(set([key.split(';')[0] for key in verzameling_categorie.keys()]))
#aantal verslagen per enkele label
print ordered_annotations.ix[enkele_labels]

X_train, Y_train, X_test, Y_test = reorderData(verzameling_categorie, enkele_labels)

clf, labels, predicted = trainAndPredict(X_train, Y_train, X_test, Y_test)

# categoriesFiles(['1MAS1', '1MAS2', '2CAL2', '3ARC1'])
#printing top 10 words that define a class, based on frequency. 
print_top10(clf.named_steps.get('vectorizer'), clf.named_steps.get('clf'), clf.classes_)











