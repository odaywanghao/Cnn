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

    if l == 1:
    	return 1.0 - np.average(np.where(preds == labels, 1, 0))
    else:
    	return 1.0 - (np.sum(np.where(preds == labels, labels, 0)) / float(N))


class PerceptronLayer():
	"""
	A perceptron layer.
	"""

	def __init__(self, no_outputs, no_inputs, outputType="relu", prob=1):
		"""
		Initialize fully connected layer.

		Args:
		-----
			no_outputs: No. output classes.
			no_inputs: No. input features.
			outputType: Type of output ('sum', 'sigmoid', 'tanh', 'relu' or 'softmax')
			prob: Prob of activation units being present during dropout i.e 1 for no dropout.
		"""
		self.o_type = outputType
		if outputType == 'sigmoid' or outputType == 'tanh':
			self.w = (6.0/(no_outputs + no_inputs)) * np.random.randn(no_outputs, no_inputs)
		else:
			self.w = 0.01 * np.random.randn(no_outputs, no_inputs)

		self.b = np.zeros((no_outputs, 1))
		self.v_w, self.v_b = 0, 0
		self.p, self.train = prob, False


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
			dEds = dEdo * sigmoid(self.s) * (1 - sigmoid(self.s))
		elif self.o_type == 'tanh':
			dEds = dEdo * sech2(self.s)
		elif self.o_type == 'relu':
			dEds = dEdo * np.where(self.s > 0, 1, 0)
		else:
			dEds = dEdo #Softmax or sum.

		self.dEdw = np.dot(dEds, self.x.T)
		self.dEdb = np.sum(dEds, axis=1).reshape(self.b.shape)
		self.dEdx = np.dot(dEds.T, self.w).T
		
		return self.dEdx


	def update(self, eps_w, eps_b, mu):
		"""
		Update the weights in this layer.

		Args:
		-----
			eps_w: Learn rate for weights.
			eps_b: Learn rate for bias.
			mu: Momentum coefficient.
		"""
		k, N = self.x.shape
		self.v_w = (self.v_w * mu) - (eps_w * (self.dEdw / N))
		self.v_b = (self.v_b * mu) - (eps_b * (self.dEdb / N))
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
		if self.train:
			self.x = data * np.random.binomial(1, self.p, data.shape)
			self.s = np.dot(self.w, self.x) + self.b
		else:
			self.x = data
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
			return self.s #Sum


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


	def train(self, train_data, train_target, valid_data, valid_target, test_data, test_target, hyperparams):
		"""
		Train the mlp on the training set and validation set using the provided
		hyperparameters.

		Args:
		----
			train_data 	:	no_instance x no_features matrix.
			train_target :	no_instance x k_class matrix.
			valid_data 	:	no_instance x no_features matrix.
			valid_target :	no_instance x k_class matrix.array shuffle numpy
			hyperparams :	A dictionary of training parameters.
		"""
		#Notify layers of training.
		for layers in self.layers:
			layers.train = True

		#Start training.
		N, m1 = train_data.shape
		N, m2 = train_target.shape

		if hyperparams['learn_rate']: #Create seperate learn rates.
			hyperparams['learn_rate_w'] = hyperparams['learn_rate']
			hyperparams['learn_rate_b'] = hyperparams['learn_rate']

		if 'momentum' not in hyperparams.keys():
			hyperparams['momentum'] = 0

		epochs = hyperparams['epochs']

		for epoch in xrange(epochs):

			self.backprop(self.predict(train_data) - train_target)
			self.update(hyperparams)

			#Measure network's performance.
			train_class = self.classify(self.predict(train_data))
			valid_class = self.classify(self.predict(valid_data))
			ce_train = mce(train_class, train_target)
			ce_valid = mce(valid_class, valid_target)
			print '\rEpoch' + "{:10.2f}".format(epoch) + ' Train MCE:' + "{:10.2f}".format(ce_train) + ' Validation MCE:' + "{:10.2f}".format(ce_valid)
			if epoch != 0 and epoch % 100 == 0:
  				print '\n'

  		#Notify layers of end of training.
  		for layers in self.layers:
			layers.train = False

		#Test time.
  		test_class = self.classify(self.predict(test_data))
		ce_test = mce(test_class, test_target)
  		print '\r Test MCE:' + "{:10.2f}".format(ce_test)


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
  			layer.update(params['learn_rate_w'], params['learn_rate_b'], params['momentum'])


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
	mlp.train(input_train.T, target_train.T, input_valid.T, target_valid.T, input_test.T, target_test.T, {'learn_rate': 0.1, 'momentum': 0.5, 'epochs': 600})


if __name__ == '__main__':

	testmlp('data/digits.npz')
