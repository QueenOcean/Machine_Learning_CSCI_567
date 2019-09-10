import numpy as np
import time
import random
from hmm import HMM


def accuracy(predict_tagging, true_tagging):
	if len(predict_tagging) != len(true_tagging):
		return 0, 0, 0
	cnt = 0
	for i in range(len(predict_tagging)):
		if predict_tagging[i] == true_tagging[i]:
			cnt += 1
	total_correct = cnt
	total_words = len(predict_tagging)
	if total_words == 0:
		return 0, 0, 0
	return total_correct, total_words, total_correct*1.0/total_words


class Dataset:

	def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
		tags = self.read_tags(tagfile)
		data = self.read_data(datafile)
		self.tags = tags
		lines = []
		for l in data:
			new_line = self.Line(l)
			if new_line.length > 0:
				lines.append(new_line)
		if seed is not None: random.seed(seed)
		random.shuffle(lines)
		train_size = int(train_test_split * len(data))
		self.train_data = lines[:train_size]
		self.test_data = lines[train_size:]
		return

	def read_data(self, filename):
		"""Read tagged sentence data"""
		with open(filename, 'r') as f:
			sentence_lines = f.read().split("\n\n")
		return sentence_lines

	def read_tags(self, filename):
		"""Read a list of word tag classes"""
		with open(filename, 'r') as f:
			tags = f.read().split("\n")
		return tags

	class Line:
		def __init__(self, line):
			words = line.split("\n")
			self.id = words[0]
			self.words = []
			self.tags = []

			for idx in range(1, len(words)):
				pair = words[idx].split("\t")
				self.words.append(pair[0])
				self.tags.append(pair[1])
			self.length = len(self.words)
			return

		def show(self):
			print(self.id)
			print(self.length)
			print(self.words)
			print(self.tags)
			return


# TODO:
def model_training(train_data, tags):
    model = None
    S = len(tags)
    
    # Find state symbols
    state_dict = dict()
    for i in range(S):
        state_dict[tags[i]] = i
        
    # Find initial probability
    sentences = len(train_data)
    pi = [0.0]*S
    for sen in range(sentences):
        ind = state_dict[train_data[sen].tags[0]]
        pi[ind] = pi[ind] + 1
    for x in range(S):
        pi[x] = pi[x]/sentences
    
    # Find obs_dict
    obs_dict = {}
    ind = 0
    for sen in range(sentences):
        sentence = train_data[sen].words
        for word in sentence:
            if word not in obs_dict.keys():
                obs_dict[word] = ind
                ind = ind +1
    
    # Find transition probabilities - A
    A = np.zeros([S, S])
    start = [0.0]*S
    for sen in range(sentences):
        sentence = train_data[sen].tags
        for i in range(len(sentence)-1):
            s = sentence[i]
            start[state_dict[s]] +=1
            sp = sentence[i+1]
            A[state_dict[s]][state_dict[sp]] += 1
    for i in range(S):
        if start[i] != 0:
            A[i] = [x/start[i] for x in A[i]]
    
    # Find emission probabilities - B
    B = np.zeros([S, ind])
    for sen in range(sentences):
        start[state_dict[train_data[sen].tags[-1]]] += 1
        for i in range(len(train_data[sen].tags)):
            s = train_data[sen].tags[i]
            o = train_data[sen].words[i]
            B[state_dict[s]][obs_dict[o]] += 1
    for i in range(S):
        if start[i] != 0:
            B[i] = [x/start[i] for x in B[i]]
    
    model = HMM(pi, A, B, obs_dict, state_dict)
    return model


# TODO:
def speech_tagging(test_data, model, tags):
    tagging = []
    L = len(test_data)
    S = len(tags)
    
    # updates for new words
    ind = max(model.obs_dict.values()) + 1
    z = np.full((S, 1), 1e-6)
    
    # Calling viterbi algorithm
    for i in range(L):
        for word in test_data[i].words:
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = ind
                model.B = np.append(model.B, z, axis=1)
                ind += 1

        tagging.append(model.viterbi(test_data[i].words))
    
    return tagging

