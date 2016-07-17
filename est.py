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

W = np.e**(2j*np.pi / N)

S1 = np.ones(20)
S2 = W ** np.arange(20)
S = np.vstack((S1, S2))

Y = np.zeros((H.shape[0], 2)) * 1j
# long and thin

# now, we go row by row here

count = 0
for row in H * x:
	res = S @ row
	Y[count] = res
	count += 1

X = np.zeros(20)

# ----

# def rem(Y, H):
# 	count = 0
# 	for y in Y:
# 		if np.all(y == 0):
# 			H[:, count] = 0
# 		count += 1

# def remSingle(H, check, variable):
# 	H[check, variable] = 0

# def peel(Y, H):
# 	# we will have many parts
# 	# ys
# 	X = np.zeros(20)
# 	rem(Y, H) # remove zero-tons
# 	while not np.all(H == 0):
# 		for y in Y:
# 			if np.all(y == 0):
# 				print('zeroton')
# 			if y[0] != 0:
# 				k = np.angle(y[1]/y[0])/ (2*np.pi/20)
# 				print(k)
# 				# k is the index for the variable node
# 				# that is the only one in this
# 				# particular check node
# 				if np.isclose(k, np.round(k)):
# 					print('singleton')
# 				result[k] = y[0]
# 				remSingle(H, check, variable)
# 				# step 2
# 				H[] = 0
# 				# step 3
# 				;
# peel(Y, H)

# We have H, Y, X
# Y check
# X variable

# H[check:variable]

def step1(H, Y):
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

def step2(H, Y, X, singletonlist, edgelist):
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

def step34(H, Y, X, singletonlist):
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

singletonlist, edgelist = step1(H, Y)
step2(H, Y, X, singletonlist, edgelist)
step34(H, Y, X, singletonlist)

# ----

# encoder

def createMatrix(N, K, R):
	H = np.zeros((R, N))
	rands = np.random.randint(0, R, 100)
	np.reshape()
	return H







