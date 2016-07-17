"""
Encode

We go from

y = S @ H @ x
"""

# 20-length
# 9 measurements
# 5-sparse

# N 20 K 5
# 9 R
# n = 9/5

import numpy as np

N = 20
K = 5
R = 9

x = np.array([0,1,0,4,0,2,0,0,0,0,3,0,0,7,0,0,0,0,0,0])
H = np.zeros((9, 20))
H[1][[1,5,13]] = 1
H[2][10] = 1
H[3][3] = 1
H[4][[5,10]] = 1
H[5][1] = 1
H[7][[3,13]] = 1


singletonlist, edgelist = step1(H, Y)
step2(H, Y, X, singletonlist, edgelist)
step34(H, Y, X, singletonlist)

# ----

# no repeats?
H = createMatrix(8, 2, 5)

# create S

class Decoder(object):
	def __init__(self, N, H):
		self.N = N
		self.H = H

	def decode(self, Y):
		self.X = np.zeros(self.N)
		singletonlist, edgelist = self.step1(H, Y)
		self.step2(self.H, Y, self.X, singletonlist, edgelist)
		self.step34(self.H, Y, self.X, singletonlist)
		return self.X

	def step1(self, H, Y):
		singletonlist = []
		edgelist = []
		i = 0 # i is index for y, or check nodes
		for y in Y:
			if y[0] != 0:
				# k is x index
				ang = np.angle(y[1]/y[0])
				k = ang/ (2*np.pi/20)
				# we must also change this
				# so it fits the unit circle we're using
				# actually, would it work?

				if np.isclose(k, np.round(k)):
					print(k)
					singletonlist.append(k)
					edgelist.append((i, k))
			i += 1
		return np.round(np.array(singletonlist)).astype(int), np.round(np.array(edgelist)).astype(int)

	def step2(self, H, Y, X, singletonlist, edgelist):
		# remove the variable and check nodes
		for k, edge in zip(singletonlist, edgelist):
			i = edge[0]
			print(k)
			X[k] = Y[i][0]
			Y[i] = 0
		# remove edges
		for k, edge in zip(singletonlist, edgelist):
			i = edge[0]
			H[i, k] = 0

	def step34(self, H, Y, X, singletonlist):
		# singletonlist is a list of
		# variable nodes that we have
		# removed then
		
		# get a list of edges connected to the removed variables
		additionalEdges = []
		Scol = np.array([1, W])
		for variable in singletonlist:
			checks = np.where(H[:, variable])[0] # list of checks for this variable
			H[:, variable] = 0
			# now get the list of checks
			# in which it fell
			# we have taken care of the redundancy
			for c in checks:
				Y[c] -= X[variable] *(Scol ** c)

class Encoder(object):
	"""docstring for Encoder"""
	def __init__(self, N, K, R):
		self.N = N # number of variable nodes
		self.K = K # sparsity
		self.R = R # number of check nodes

	def encode(self, X):
		H = createH()
		S = createS()
		Y = np.zeros((H.shape[0], 2)) * 1j

		count = 0
		for row in H * X:
			res = S @ row
			# Python 3.5 syntax
			Y[count] = res
			count += 1
		return Y

	def createH(self, N, K, R):
		H = np.zeros((R, N), dtype=np.int)
		rands = np.zeros((N, K), dtype=np.int)
		# fill rands
		count = 0
		for row in rands:
			r = np.random.choice(R, K, replace=False)
			rands[count] = r
			count += 1

		variable_from = 0
		# visualize how we are traversing
		for row in rands: 
			# how many in each row?
			for i in range(K):
				check_to = row[i]
				H[check_to, variable_from] = 1
			variable_from += 1
		return H

	def createS(self, N):
		W = np.e**(2j*np.pi / N)
		S1 = np.ones(N)
		S2 = W ** np.arange(N)
		first = np.vstack((S1, S2))
		second = np.random.normal(0, 1, N)
		return first * second
