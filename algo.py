import numpy as np

class Decoder(object):
	"""docstring for Decoder"""
	def __init__(self, data, N):
		self.data = data
		self.N = N
	def peel(self):
		# check for all singletons
		for x, y in data:
			if :

class Encoder(object):
	"""docstring for Encoder"""
	def __init__(self, N, K, ):
		self.N = N # the higher dimensionality
		self.K = K # sparsity

	def encode(self, data):
		createH()
		createS()

	def createH(self):
		self.H

	def createS(self):
		w = np.e**(2j*np.pi)/self.N
		wvec = w ** np.arange(self.N)
		self.S = np.vstack((np.ones(self.N), wvec))



