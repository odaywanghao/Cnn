# Copyright (c) 2015 ev0, lautimothy
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
from copy import deepcopy
from util import *


def sigmoid(data):
	"""
	Run The sigmoid activation function over the input data.

	Args:
	----
		data : A k x N array.

	Returns:
	-------
		A k x N array.
	"""
	return 1 / (1 + np.exp(-data))


def softmax(data):
	"""
	Run the softmax activation function over the input data.

	Args:
	----
		data : A k x N array.

	Returns:
	-------
		A k x N array.
	"""
	k, N = data.shape
	e = np.exp(data)
	return e/np.sum(e, axis=0).reshape(1, N)


def sech2(data):
	"""
	Find the hyperbolic secant function over the input data.

	Args:
	-----
		data : A k x N array.

	Returns:
	--------
		A k x N array.
	"""
	return np.square(1 / np.cosh(data))


def relu(data):
	"""
	Perform rectilinear activation on the data.

	Args:
	-----
		data: A k x N array.

	Returns:
	--------
		A k x N array.
	"""
	return np.maximum(data, 0)


def epsilon_decay(eps, phi, satr, itr, intvl):
	"""
	Decay the given learn rate given.

	Args:
	-----
		eps: Learning rate.
		phi: Learning decay.
		satr: Iteration to saturate learning rate or string 'Inf'.
		itr: Current iteration.
		intvl: Decay interval i.e 0 (constant), 1 (progressive) etc.

	Returns:
	--------
		The learning rate to apply.
	"""
	if intvl != 0:
		i = min(itr, float(satr)) / intvl
		return eps / (1.0 + (i * phi))
	else:
		return eps


def cross_entropy(preds, labels):
    """
    Compute the cross entropy over the predictions.

    Args:
    -----
        preds : An N x k array of class predictions.
        labels : An N x k array of class labels.
    
    Returns:
    --------
        The cross entropy.
    """
    N, k = preds.shape

    if k == 1: #sigmoid
        return -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
    
    return -np.sum(np.sum(labels * np.log(preds), axis=1).reshape(N, 1))


def mce(preds, labels):
    """
    Compute the mean classification error over the predictions.

    Args:
    ----
        preds, labels : An N x k binary array or an N x 1 array of class predictions.
    
    Returns:
    --------
        The mean classification error over the predictions.
    """
    N, l = labels.shape

    if preds.shape != labels.shape:
    	print "Mce Error: Inputs unequal."
    	return None

    if l == 1:
    	return 1.0 - np.average(np.where(preds == labels, 1, 0))
    else:
    	return 1.0 - (np.sum(np.where(preds == labels, labels, 0)) / float(N))


class PerceptronLayer():
	"""
	A perceptron layer.
	"""

	def __init__(self, no_outputs, no_inputs, prob=1, outputType='relu', init_w=0.01, init_b=0):
		"""
		Initialize fully connected layer.

		Args:
		-----
			no_outputs: No. output classes.
			no_inputs: No. input features.
			prob: Prob. of a neuron being present during dropout.
			outputType: String repr. type of output i.e 'linear', 'sigmoid', 'tanh', 'relu' or 'softmax'.
			init_w: Std dev of initial weights drawn from a std Normal distro.
			init_b: Initial value of biases.
		"""
		self.o_type = outputType
		self.init_w, self.init_b = init_w, init_b
		
		if outputType == 'sigmoid' or outputType == 'tanh':
			self.init_w = (6.0 / (no_outputs + no_inputs))

		self.w = self.init_w * np.random.randn(no_outputs, no_inputs)
		self.b = self.init_b * np.ones((no_outputs, 1))
		self.p, self.train = prob, False
		self.v_w, self.dw_ms, self.v_b, self.db_ms = 0, 0, 0, 0


	def bprop(self, dEdo):
		"""
		Compute gradients and return sum of error from output down
		to this layer.

		Args:
		-----
			dEdo: A no_output x N array of errors from prev layers.

		Return:
		-------
			A no_inputs x N array of input errors.
		"""
		if self.o_type == 'sigmoid':
			dods = sigmoid(self.s) * (1 - sigmoid(self.s))
			dEds = dEdo * dods
		elif self.o_type == 'tanh':
			dods = sech2(self.s)
			dEds = dEdo * dods
		elif self.o_type == 'relu':
			dods = np.where(self.s > 0, 1, 0)
			dEds = dEdo * dods
		else:
			dEds = dEdo #Softmax or sum.

		x = (self.x * self.dropped)
		self.dEdw = np.dot(dEds, x.T) / dEdo.shape[1]
		self.dEdb = np.sum(dEds, axis=1).reshape(self.b.shape) / dEdo.shape[1]
		return np.dot(dEds.T, self.w).T * self.dropped #dEdx


	def update(self, eps_w, eps_b, mu, l2, useRMSProp, RMSProp_decay, minsq_RMSProp):
		"""
		Update the weights in this layer.

		Args:
		-----
			eps_w, eps_b: Learning rates for the weights and biases.
			mu: Momentum coefficient.
			l2: L2 regularization coefficent.
			useRMSProp: Boolean indicating the use of RMSProp.
			RMSProp_decay: Decay term for the squared average.
			minsq_RMSProp: Constant added to square-root of squared average. 
		"""
		if useRMSProp:
			self.dw_ms = (RMSProp_decay * self.dw_ms) + ((1.0 - RMSProp_decay) * np.square(self.dEdw))
			self.db_ms = (RMSProp_decay * self.db_ms) + ((1.0 - RMSProp_decay) * np.square(self.dEdb))
			self.dEdw = self.dEdw / (np.sqrt(self.dw_ms) + minsq_RMSProp)
			self.dEdb = self.dEdb / (np.sqrt(self.db_ms) + minsq_RMSProp)
			self.dEdw[np.where(np.isnan(self.dEdw))] = 0
			self.dEdb[np.where(np.isnan(self.dEdb))] = 0

		self.v_w = (mu * self.v_w) - (eps_w * self.dEdw) - (eps_w * l2 * self.w)
		self.v_b = (mu * self.v_b) - (eps_b * self.dEdb) - (eps_b * l2 * self.b)
		self.w = self.w + self.v_w
		self.b = self.b + self.v_b


	def feedf(self, data):
		"""
		Perform a forward pass on the input data.

		Args:
		-----
			data: An no_inputs x N array.

		Return:
		-------
			A no_outputs x N array.
		"""
		self.x = data

		if self.train:
			self.dropped = np.random.binomial(1, self.p, data.shape)
			self.s = np.dot(self.w, self.x * self.dropped) + self.b
		else:
			self.s = np.dot(self.w * self.p, self.x) + self.b

		if self.o_type == 'sigmoid':
			return sigmoid(s)
		elif self.o_type == 'tanh':
			return np.tanh(self.s)
		elif self.o_type == 'relu':
			return relu(self.s)
		elif self.o_type == 'softmax': 
			return softmax(self.s)
		else:
			return self.s #linear


class Mlp():
	"""
	A multilayer perceptron.
	"""

	def __init__(self, layers):
		"""
		Initialize the mlp.

		Args:
		-----
			layers: List of mlp layers arranged heirarchically.
		"""
		self.layers = deepcopy(layers)


	def train(self, train_data, train_target, valid_data, valid_target, test_data, test_target, hyperparameters):
		"""
		Train the mlp on the training set and validation set using the provided
		hyperparameters.

		Args:
		----
			train_data 	:	no_instance x no_features matrix.
			train_target :	no_instance x k_class matrix.
			valid_data 	:	no_instance x no_features matrix.
			valid_target :	no_instance x k_class matrix.array shuffle numpy
			hyperparameters :	A dictionary of training parameters.
		"""
		N, m1 = train_data.shape
		N, m2 = train_target.shape

		# Train the network with batch 'cos
		# online too erratic & mini-batch too much work.
		epochs = hyperparameters['epochs']

		for epoch in xrange(epochs):

			self.backprop(self.predict(train_data) - train_target)
			self.update(hyperparameters)

			#Measure network's performance.
			train_class = self.classify(self.predict(train_data))
			valid_class = self.classify(self.predict(valid_data))
			ce_train = mce(train_class, train_target)
			ce_valid = mce(valid_class, valid_target)
			print '\rEpoch' + "{:10.2f}".format(epoch) + ' Train MCE:' + "{:10.2f}".format(ce_train) + ' Validation MCE:' + "{:10.2f}".format(ce_valid)
			if epoch != 0 and epoch % 100 == 0:
  				print '\n'

  		test_class = self.classify(self.predict(test_data))
		ce_test = mce(test_class, test_target)
  		print '\r Test MCE:' + "{:10.2f}".format(ce_test)

  		return 0


  	def backprop(self, dEds):
  		"""
  		Propagate the error gradients through the network.

  		Args:
  		-----
  			dEds: Error gradient w.r.t WX.
  		"""
  		error = dEds.T
  		for i in xrange(len(self.layers)):
  			error = self.layers[i].bprop(error)


  	def update(self, params):
  		"""
  		Update the network weights using the training
  		parameters.

  		Args:
  		-----
  			params: Training parameters.
  		"""
  		for layer in self.layers:
  			layer.update(params['eps_w'], params['eps_b'], params['mu'], params['l2'], params['RMSProp'], params['RMSProp_decay'], params['minsq_RMSProp'])


  	def predict(self, data):
  		"""
  		Return the posterior probability distribution of data
  		over k classes.

  		Args:
  		-----
  			data: An N x k array of input data.

  		Returns:
  		-------
  			An N x k array of posterior distributions.
  		"""
  		x = data.T
  		for i in xrange(len(self.layers) - 1, 0, -1):
  			x = self.layers[i].feedf(x)

  		return self.layers[0].feedf(x).T


  	def classify(self, prediction):
		"""
		Peform classification using the class predictions of the classifier.
		"""
		m, n = prediction.shape

		if n == 1:
			for row in xrange(m):
				if prediction[row, 0] < 0.5:
					prediction[row, 0] = 0
				else:
					prediction[row, 0] = 1 
		else:
			for row in xrange(m):
				i = 0
				for col in xrange(n):
					if prediction[row, col] > prediction[row, i]:
						i = col
				prediction[row, :] = np.zeros(n)
				prediction[row, i] = 1

		return prediction


	def saveModel(self):
		"""
		Save the neural network model.
		"""
		#TODO: Implement.
		pass


	def loadModel(self, model):
		"""
		Load a model for this neural network.
		"""
		#TODO: Implement this later.
		pass


def testmlp(filename):
  	"""
  	Test mlp with mnist 2 and 3 digits.

  	Args:
  	-----
    	filename: Name of the file for 2 and 3.
  	"""
	data = np.load(filename)
	input_train = np.hstack((data['train2'], data['train3']))
	input_valid = np.hstack((data['valid2'], data['valid3']))
	input_test = np.hstack((data['test2'], data['test3']))
	target_train = np.hstack((np.zeros((1, data['train2'].shape[1])), np.ones((1, data['train3'].shape[1]))))
	target_valid = np.hstack((np.zeros((1, data['valid2'].shape[1])), np.ones((1, data['valid3'].shape[1]))))
	target_test = np.hstack((np.zeros((1, data['test2'].shape[1])), np.ones((1, data['test3'].shape[1]))))

	mlp = Mlp([PerceptronLayer(1, 10), PerceptronLayer(10, 256, "tanh")])
	mlp.train(input_train.T, target_train.T, input_valid.T, target_valid.T, input_test.T, target_test.T, {'eps_w': 0.1, 'eps_b': 0.1, 'mu': 0.9, 'l2': 0, 'epochs': 1600})


if __name__ == '__main__':

	testmlp('data/digits.npz')
