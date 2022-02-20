from __future__ import division
import pickle
import argparse
from tkinter import SEL
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from sklearn import metrics
# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

__authors__ = ['Jean-Marc D','Aspram Grigoryan','Felix Hans', 'Patrick Leyendecker']
__emails__  = ['jean-marc.dossou-yovo@outlook.fr','aspram.grigoryan@student-cs.fr','b00782416@essec.edu', 'b00784830@essec.edu']

def text2sentences(path):
	# feel free to make a better tokenization/pre-processing
	sentences = []
	with open(path) as f:
		for l in f: 
			sentences.append(list(map(lambda x : re.sub("\W+","",x),l.lower().split())) )
	sentences = [e for e in sentences if e != []]
	return sentences

def loadPairs(path):
	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs


class SkipGram:
	def __init__(self, sentences, nEmbed=100, negativeRate=10, winSize = 5, minCount = 6):
		self.winSize = winSize
		self.negativeRate = negativeRate
		self.w2id = {} # word to ID mapping
		self.trainset = sentences# set of sentences
		self.vocab = []
		counter = defaultdict(lambda : 0)
		unigram = []
		for i,sen in enumerate(sentences) :
			for w in sen:
				counter[w] += 1
				if counter[w] == minCount :
					self.vocab.append(w)

		self.w2id = dict(map(lambda t: (t[1], t[0]), enumerate(self.vocab)))
		self.learning_rate = 0.08

		#Creating unigram distribution for negative sampling
		z = 0
		for w in self.vocab : 
			unigram.append(counter[w])
			z += counter[w]
			pass
		unigram = np.array(unigram) / z
		alpha = 0.6
		unigram = np.power(unigram,alpha)
		unigram = unigram / sum(unigram)
		self.unigram = dict(map(lambda x :(x[0],x[1]) ,enumerate(unigram)))
		self.nEmbed =nEmbed
		self.W = np.random.uniform(size=(len(self.vocab),nEmbed))
		self.C = np.random.uniform(size=(len(self.vocab),nEmbed))
		self.trainWords = 0
		self.loss =[]


	def sample(self, omit):
		filtered  = {key: value for key, value in self.unigram.items() if key not in omit}
		p = np.array(list(filtered.values())) 
		p = p / sum(p)
		negative_ids = np.random.choice( list(filtered.keys()), size = self.negativeRate, p=p)
		return negative_ids

	
	def feed_forward(self,wordId, contextId, negativeIds):
		x = np.zeros((len(self.vocab),1))
		x[wordId,0] = 1
		
		self.intermediate = (x.T @ self.W).T
		out  = (self.C @ self.intermediate)[np.concatenate(([contextId],negativeIds),axis=None),0]
		self.out = 1/(1 + np.exp(-out))
		

	def	backpropagate(self,wordId, contextId, negativeIds) :
		y = np.zeros((len(self.out),))
		y[0] = 1
		error = self.out - y
		self.accLoss += sum(np.power(error,2))
		error = error.reshape((len(self.out),1))
		d_1 = error.T @ self.C[np.concatenate(([contextId],negativeIds),axis=None),:]
		d_2 = np.dot(error , self.intermediate.T)
		d_1 = d_1.reshape((self.nEmbed,))
		self.W[wordId,:] += - self.learning_rate * d_1
		self.C[np.concatenate(([contextId],negativeIds),axis=None),:] += - self.learning_rate * d_2

		
	def train(self,epochs=10):
		for a in range(epochs) : 
			self.accLoss = 0
			self.trainWords = 0
			for counter, sentence in enumerate(self.trainset):
				sentence = list(filter(lambda word: word in self.vocab, sentence))

				for wpos, word in enumerate(sentence):
					wIdx = self.w2id[word]
					winsize = np.random.randint(self.winSize) + 1
					start = max(0, wpos - winsize)
					end = min(wpos + winsize + 1, len(sentence))
					for context_word in sentence[start:end]:
						ctxtId = self.w2id[context_word]
						if ctxtId == wIdx: continue
						negativeIds = self.sample({wIdx, ctxtId})
						self.trainWord(wIdx, ctxtId, negativeIds)
						self.trainWords += 1

			print('Epoch ', a +1)
			print('Loss ', self.accLoss/self.trainWords)
		

	def trainWord(self, wordId, contextId, negativeIds):
		self.feed_forward(wordId, contextId, negativeIds)
		self.backpropagate(wordId, contextId, negativeIds)

	def save(self,path):
                sample_list = [self.W, self.vocab, self.w2id]
                open_file = open(path, "wb")
                pickle.dump(sample_list, open_file)
                open_file.close()

	def similarity(self,word1,word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
		#input (word1,word2) must be a 2D-vector
		if word1 in self.vocab :
			word1 = self.W[self.w2id[word1],:]
		else : 
			word1 = np.ones((self.W.shape[1],))/2

		if word2 in self.vocab :
			word2 = self.W[self.w2id[word2],:]
		else : 
			word2 = np.ones((self.W.shape[1],))/2
		
		similarity =np.inner(word1,word2)/(np.linalg.norm(word1)* np.linalg.norm(word2))

		return similarity
		
	@staticmethod
	def load(path):
		model = SkipGram([])
		open_file = open(path, "rb")
		loaded_list = pickle.load(open_file)
		open_file.close()
		model.W = loaded_list[0]
		model.vocab = loaded_list[1]
		model.w2id = loaded_list[2]
		return model


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences)
		sg.train()
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))

