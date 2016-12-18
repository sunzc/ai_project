#!/usr/bin/env python3
# Project : SMS Spam Classifier
# Author: Zhichuang Sun
# Data: 12/15/2016

import re
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

class SMSSpamCla:
	"""
	data path:  ./data/SMSSpamCollection
	labeled data follow the following format:

	spam	You won the verizon lucky ball, and get a bonus of $10000
	ham	Do you have time this afternoon?
	...

	we should split into two part: the first 30% for training, the other for testing.
	"""
	def __init__(self, train_rate, data_path):
		self.train_data, self.train_target, self.test_data, self.test_target = self.extract_data_target(data_path, train_rate)
		self.target_names = ['spam','ham']

	def get_train_vector(self, percent, token_pat):
		self.bound = int(len(self.train_data)/100) * percent

		self.count_vect = CountVectorizer(token_pattern = token_pat)
		self.tokenizer = self.count_vect.build_tokenizer()
		self.X = self.count_vect.fit_transform(self.train_data[:self.bound])
		self.Y = self.train_target[:self.bound]

	def preprocess(self, msg):
		# replace numbers with 'N' to preserve the pattern of phone numbers
		return re.sub(r'\d',r'N',msg)

	def extract_data_target(self, data_path, train_rate):
		corpus = open(data_path, 'r').readlines()
		total_number = len(corpus)
		train_number = int(total_number * train_rate)

		data = []
		# spam: -1, ham: 1
		target = []
		for line in corpus:
			fields = line.split('\t')
			if fields[0] == 'spam':
				target.append(-1)
			elif fields[0] == 'ham':
				target.append(1)
			else:
				print("unsupported data format:" + line)
				continue

			if (len(fields) == 2):
				cooked_msg = self.preprocess(fields[1])
				data.append(cooked_msg)
			else:
				msg = ""
				for i in range(1,len(fields)):
					msg += fields[i]
					print(fields[i])
				cooked_msg = self.preprocess(msg)
				data.append(cooked_msg)

		shuffled_idx = list(range(len(data)))
		random.shuffle(shuffled_idx)
		sf_data = []
		sf_target = []
		for i in shuffled_idx:
			sf_data.append(data[i])
			sf_target.append(target[i])

		train_data = sf_data[:train_number]
		train_target = sf_target[:train_number]
		test_data = sf_data[train_number:]
		test_target = sf_target[train_number:]

		return train_data, train_target, test_data, test_target

	def train_mnb(self):
		self.cla = MultinomialNB().fit(self.X, self.Y)

	def train_gnb(self):
		self.cla = GaussianNB().fit(self.X.toarray(), self.Y)

	def binary_array(self, arr):
		xx = arr
		for x in range(len(xx)):
			for y in range(len(xx[x])):
				if xx[x][y] != 0:
					xx[x][y] = 1
		return xx

	def train_bnb(self):
		self.cla = BernoulliNB().fit(self.binary_array(self.X.toarray()), self.Y)

	def train_svm(self):
		self.cla = SVC(kernel = 'linear', class_weight = 'balanced').fit(self.X, self.Y)

	def train_neighbor(self):
		self.cla = KNeighborsClassifier(n_neighbors = self.K).fit(self.X, self.Y)

	def train_tree(self):
		self.cla = DecisionTreeClassifier().fit(self.X, self.Y)

	def train_ensemble(self):
		self.cla = BaggingClassifier(KNeighborsClassifier(n_neighbors = 1), max_samples = 0.5, max_features = 0.5).fit(self.X, self.Y)
		#self.cla = BaggingClassifier(SVC(kernel = 'linear', class_weight = 'balanced'), max_samples = 0.5, max_features = 0.5).fit(self.X, self.Y)

	def train(self, model_id, k):
		self.model_id = model_id
		self.models = {0:self.train_mnb, 1:self.train_gnb, 2:self.train_bnb, 3:self.train_svm, 4:self.train_neighbor, 5:self.train_tree, 6:self.train_ensemble}
		self.model_names = {0:"MultinomialNB",1:"GaussianNB", 2:"BernoulliNB", 3:"Support Vector Machine", 4:"KNeighborsClassifier", 5:"DecisionTreeClassifier", 6:"Ensemble Bagging KNeighborsClassifier"}

		self.K = k
		self.models[model_id]()

		return self.model_names[model_id]

	def spam_caught(self, y_true, y_pred):
		spam_total = 0
		spam_caught = 0
		for i in range(len(y_true)):
			if y_true[i] == -1:
				spam_total += 1
				if y_pred[i] == -1:
					spam_caught += 1
		print("Spam Total: " + str(spam_total) + " Spam Caught: "+ str(spam_caught))
		return spam_caught/spam_total

	def block_ham(self, y_true, y_pred):
		ham_total = 0
		ham_block = 0
		for i in range(len(y_true)):
			if y_true[i] == 1:
				ham_total += 1
				if y_pred[i] == -1:
					ham_block += 1
		print("Ham Total: " + str(ham_total) + " Ham Blocked: "+ str(ham_block))
		return ham_block/ham_total

	def accuracy(self, y_true, y_pred):
		total = len(y_true)
		accurate = 0
		for i in range(len(y_true)):
			if y_true[i] == y_pred[i]:
				accurate += 1
		print("Total: " + str(total) + " accurate predict: "+ str(accurate))
		return accurate/total


	def get_result(self):
		result_array = []

		X_test = self.count_vect.transform(self.test_data)
		if self.model_id == 1:
			# Gaussian NB require dense array
			prediction = self.cla.predict(X_test.toarray())
		elif self.model_id == 2:
			# Bernoulli NB require binary value
			prediction = self.cla.predict(self.binary_array(X_test.toarray()))
		else:
			prediction = self.cla.predict(X_test)
		print(metrics.classification_report(self.test_target, prediction, target_names=self.target_names))

		mcc = matthews_corrcoef(self.test_target, prediction)
		acc = self.accuracy(self.test_target, prediction)
		sc = self.spam_caught(self.test_target, prediction)
		bh = self.block_ham(self.test_target, prediction)

		print("matthews_corrcoef:" + str(mcc))
		print("Acc:" + str(acc))
		print("SC:" + str(sc))
		print("BH:" + str(bh))

		result_array.append(sc)
		result_array.append(bh)
		result_array.append(acc)
		result_array.append(mcc)

		return result_array

	def predict(self, msg):
		test_data = []
		test_data.append(self.preprocess(msg))
		X_test = self.count_vect.transform(test_data)

		if self.model_id == 1:
			# Gaussian NB require dense array
			prediction = self.cla.predict(X_test.toarray())
		elif self.model_id == 2:
			# Bernoulli NB require binary value
			prediction = self.cla.predict(self.binary_array(X_test.toarray()))
		else:
			prediction = self.cla.predict(X_test)

		if prediction[0] == -1:
			#print("spam : " + msg)
			print("spam")
		else:
			print("ham")

if __name__ == "__main__":
	data_path = './data/SMSSpamCollection'
	tok1 = u'(?u)\\b\\w+\\w*\\b'
	tok2 = u'(?u)\\b\\w+[\\-\\.\\,\\:]*\\w*\\b'
	results = []
	for K in range(1,10):
		cla = SMSSpamCla(0.3, data_path)
		cla.get_train_vector(100, tok1)
		model_name = cla.train(4, K)
		res = cla.get_result()
		label = model_name + "+" + "tok1,k="+str(K)
		results.append((label,res, cla))

	sorted_results = sorted(results, key=lambda record: (1 - record[1][3]))
	print("Classifier "+"\t\t\t\t\tSC%\tBH%\tAcc%\tMCC")
	for x in sorted_results:
		tabs = int(len(x[0])/8)
		print (x[0] + ((6-tabs) * "\t") + "%1.3f\t%1.3f\t%1.3f\t%1.3f" %(x[1][0], x[1][1], x[1][2], x[1][3]))

#	tok_pats = [tok1, tok2]
#	train_rates = [x*0.1 for x in range(3,4)]
#	cla_nums = 7
#
#	model_name = ""
#	results = []
#	for i in range(cla_nums):
#		for j in range(len(tok_pats)):
#			cla = SMSSpamCla(0.3, data_path)
#			cla.get_train_vector(100, tok_pats[j])
#			model_name = cla.train(i)
#			print("Results: %s ,tok_pats:%s" % (model_name, tok_pats[j]))
#			res = cla.get_result()
#			label = model_name + "+" + "tok"+str(j+1)
#			results.append((label,res, cla))
#			print("")
#
#	sorted_results = sorted(results, key=lambda record: (1 - record[1][3]))
#	print("Classifier "+"\t\t\t\t\tSC%\tBH%\tAcc%\tMCC")
#	for x in sorted_results:
#		tabs = int(len(x[0])/8)
#		print (x[0] + ((6-tabs) * "\t") + "%1.3f\t%1.3f\t%1.3f\t%1.3f" %(x[1][0], x[1][1], x[1][2], x[1][3]))
#
#	print("")
#
#	msg = "test"
#	while msg != "":
#		msg = input('Enter a SMS:')
#		for x in sorted_results:
#			cla = x[2]
#			print(x[0]+"(Acc:%1.3f)"%(x[1][2]))
#			cla.predict(msg)
