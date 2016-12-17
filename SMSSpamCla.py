#!/usr/bin/env python3
# Project : SMS Spam Classifier
# Author: Zhichuang Sun
# Data: 12/15/2016

import re
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

	def train_nb(self):
		self.cla = MultinomialNB().fit(self.X, self.Y)

	def train_svm(self):
		self.cla = SVC(kernel = 'linear', class_weight = 'balanced').fit(self.X, self.Y)

	def spam_caught(self, y_true, y_pred):
		spam_total = 0
		spam_caught = 0
		for i in range(len(y_true)):
			if y_true[i] == -1:
				spam_total += 1
				if y_pred[i] == -1:
					spam_caught += 1
		return spam_caught/spam_total

	def block_ham(self, y_true, y_pred):
		ham_total = 0
		ham_block = 0
		for i in range(len(y_true)):
			if y_true[i] == 1:
				ham_total += 1
				if y_pred[i] == -1:
					ham_block += 1
		return ham_block/ham_total

	def get_result(self):
		X_test = self.count_vect.transform(self.test_data)
		prediction = self.cla.predict(X_test)
		print(metrics.classification_report(self.test_target, prediction, target_names=self.target_names))
		print("matthews_corrcoef:" + str(matthews_corrcoef(self.test_target, prediction)))
		print("SC:" + str(self.spam_caught(self.test_target, prediction)))
		print("BH:" + str(self.block_ham(self.test_target, prediction)))

	def predict(self, msg):
		test_data = []
		test_data.append(self.preprocess(msg))
		X_test = self.count_vect.transform(test_data)
		prediction = self.cla.predict(X_test)
		if prediction[0] == -1:
			print("spam : " + msg)
		else:
			print("ham : " + msg)

if __name__ == "__main__":
	tok1 = u'(?u)\\b\\w+\\w*\\b'
	tok2 = u'(?u)\\b\\w+[\\-\\.\\,\\:]*\\w*\\b'

	print("Result by cla1: using tok1:"+tok1)
	cla1 = SMSSpamCla(0.3, './data/SMSSpamCollection')
	cla1.get_train_vector(100, tok1)
	#cla.train_nb()
	cla1.train_svm()
	cla1.get_result()

	print("Result by cla2: using tok2:"+tok2)
	cla2 = SMSSpamCla(0.3, './data/SMSSpamCollection')
	cla2.get_train_vector(100, tok2)
	#cla.train_nb()
	cla2.train_svm()
	cla2.get_result()

	msg = "test"
	while msg != "":
		msg = input('Enter a SMS:')
		print("Result by cla1:")
		print(cla1.tokenizer(msg))
		cla1.predict(msg)

		print("Result by cla2:")
		print(cla2.tokenizer(msg))
		cla2.predict(msg)
