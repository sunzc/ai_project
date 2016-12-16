#!/usr/bin/env python3
# Project : SMS Spam Classifier
# Author: Zhichuang Sun
# Data: 12/15/2016

import random

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
				data.append(fields[1])
			else:
				msg = ""
				for i in range(1,len(fields)):
					msg += fields[i]
					print(fields[i])
				data.append(msg)

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

	def predict(self, msg):
		# TODO
		pass

if __name__ == "__main__":
	cla = SMSSpamCla(0.3, './data/SMSSpamCollection')
	print(len(cla.train_data))
	print(len(cla.train_target))
	print(len(cla.test_data))
	print(len(cla.test_target))
