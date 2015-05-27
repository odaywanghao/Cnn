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
import theano as thn
import theano.tensor as tn
import theano.tensor.nnet.conv as conv
from skimage.transform import downscale_local_mean as downsample
from copy import deepcopy
from mlp import *
from util import *


def upsample(data, factor):
	"""
	Upsample the errors for data by a given factor.

	Args:
	-----
		data: An N x l x m1 x n1 array of feature maps.
		factor: Upsampling factor.

	Returns:
	--------
		An N x l x m2 x n2 array of feature maps.
	"""
	return np.kron(data, np.ones(factor)) * (1.0 / np.prod(factor))


def fastConv2d(data, kernel, convtype='valid'):
	"""
	Convolve data with the given kernel.

	Args:
	-----
		data: A N x l x m2 x n2 array.
		kernel: An k x l x m1 x n1 array.

	Returns:
	--------
		A N x k x m x n array representing the output.
	"""
	d = tn.dtensor4('d')
	k = tn.dtensor4('k')
	f = thn.function([d, k], conv.conv2d(d, k, data.shape, kernel.shape, convtype))
	return f(data, kernel)


def rot2d90(data, no_rots):
	"""
	Rotate the 2d planes in a 4d array by 90 degrees no_rots times.

	Args:
	-----
		data: A N x k x m x n array.
		no_rots: An integer repr. the no. rotations by 90 degrees.

	Returns:
	--------
		A N x k x m x n array with each m x n plane rotated.
	"""
	#Stack, cut & place, rotate, cut & place, break.
	N, k, m, n = data.shape
	result = data.reshape(N * k, m, n)
	result = np.transpose(result, (2, 1, 0))
	result = np.rot90(result, no_rots)
	result = np.transpose(result, (2, 1, 0))
	result = result.reshape(N, k, m, n)

	return result



class ConvLayer():
	"""
	A Convolutional layer.
	"""

	def __init__(self, k, l, ksize, pfctr):
		"""
		Initialize convolutional layer.

		Args:
		-----
			k: No. feature maps in layer.
			l: No. input planes in layer or no. channels in input image.
			ksize: Tuple repr. the size of a kernel.
			pfctr: Pooling/subsampling factor.
		"""
		self.pfctr = pfctr
		self.kernels = (6.0 / (k * np.prod(ksize))) * np.random.randn(k, l, ksize[0], ksize[1])
		self.bias = np.zeros((k, 1, 1))


	def bprop(self, dE):
		"""
		Compute error gradients and return sum of error from output down
		to this layer.

		Args:
		-----
			dE: A N x k x m2 x n2 array of errors from prev layers.

		Returns:
		-------
			A N x l x m1 x n1 array of errors.
		"""
		dEds = upsample(dE, self.pfctr) * sech2(self.maps + self.bias)

		self.dEdb = np.sum(np.sum(np.average(dEds, axis=0), axis=1), axis=1).reshape(self.bias.shape)

		#Correlate.
		xs = rot2d90(self.x, 2)
		xs, dEds = np.swapaxes(xs, 0, 1), np.swapaxes(dEds, 0, 1)
		self.dEdw = fastConv2d(xs, dEds) / N

		#Correlate
		kernels = rot2d90(self.kernels, 2)
		dEds, kernels = np.swapaxes(dEds, 0, 1), np.swapaxes(kernels, 0, 1)
		self.dEdx = fastConv2d(dEds, kernels, 'full')

		return self.dEdx


	def update(self, lr):
		"""
		Update the weights in this layer.

		Args:
		-----
			lr: Learning rate.
		"""
		self.kernels = self.kernels - (lr * self.dEdw)
		self.bias = self.bias - (lr * self.dEdb)


	def feedf(self, data):
		"""
		Perform a forward pass on the input data.

		Args:
		-----
			data: An N x l x m1 x n1 array of input plains.

		Returns:
		-------
			A N x k x m2 x n2 array of output plains.
		"""
		self.x = data
		N, l, m1, n1 = self.x.shape
		self.maps = fastConv2d(self.x, self.kernels)
		return downsample(np.tanh(self.maps + self.bias), (1, 1, self.pfctr[0], self.pfctr[1]))


class Cnn():
	"""
	Convolutional neural network class.
	"""

	def __init__(self, layers):
		"""
		Initialize network.

		Args:
		-----
			layers: Dict. of fully connected and convolutional layers arranged heirarchically.
		"""
		self.layers = deepcopy(layers["fully-connected"]) + deepcopy(layers["convolutional"])
		self.div_x_shape = None
		self.div_ind = len(layers["fully-connected"])


	def train(self, train_data, train_label, valid_data, valid_label, test_data, test_label, params):
		"""
		Train the convolutional neural net on the training and validation set.

		Args:
		-----
			train_data 	:	no_imgs x img_length x img_width x no_channels array of images.
			valid_data 	:	no_imgs x img_length x img_width x no_channels array of images.
			test_data	:	no_imgs x img_length x img_width x no_channels array of images.
			train_label :	k x no_imgs binary array of image class labels.
			valid_label :	k x no_imgs binary array of image class labels.
			test_label	:	k x no_imgs binary array of image class labels.
			params 		:	A dictionary of training parameters.
		"""

		for i in xrange(200):

			pred = self.predict(train_data)
			label = train_label

			self.backprop(pred - label) #Start differentation. Note: dw = w - lr * dEdw.
			self.update(params)

			train_clfn = self.classify(pred)
			valid_clfn = self.classify(self.predict(valid_data))

			train_ce, valid_ce = mce(train_clfn, label), mce(valid_clfn, valid_label)

			print '\rIteration:' + "{:10.2f}".format(i) + ' Train MCE:' + "{:10.2f}".format(train_ce) + ' Valid MCE:' + "{:10.2f}".format(valid_ce)
			if i != 0 and i % 100 == 0:
  				print '\n'

  		test_clfn = self.classify(self.predict(test_data))
  		test_ce = mce(test_clfn, test_label)
  		print '\rTest MCE:' + "{:10.2f}".format(test_ce)


	def backprop(self, dE):
		"""
		Propagate the error gradients through the network.

		Args:
		-----
			dE: A no_imgs x k_classes array of error gradients.
		"""
		error = dE.T
		for layer in self.layers[0 : self.div_ind]:
			error = layer.bprop(error)

		N, k, m, n = self.div_x_shape #Reshape output from fully-connected layer.
		error = error.T.reshape(N, k, m, n)

		for layer in self.layers[self.div_ind:]:
			error = layer.bprop(error)


	def predict(self, imgs):
		"""
		Return the probability distribution over the class labels for
		the given images.

		Args:
		-----
			imgs: A no_imgs x img_length x img_width x img_channels array.

		Returns:
		-------
			A no_imgs x k_classes array.
		"""
		#Transform 4d array into N, no.input_planes, img_length, img_width array
		data = np.transpose(imgs, (0, 3, 1, 2))

		for i in xrange(len(self.layers) - 1, self.div_ind - 1, -1):
			data = self.layers[i].feedf(data)

		self.div_x_shape = data.shape #Reshape output of convolutional layer.
		data = data.reshape(data.shape[0], np.prod(data.shape[1:])).T

		for i in xrange(self.div_ind - 1, -1, -1):
			data = self.layers[i].feedf(data)

		return data.T


	def update(self, params):
		"""
		Update the network weights.

		Args:
		-----
			params: Training parameters.
		"""
		for layer in self.layers:
			layer.update(params['learn_rate'])


	def classify(self, prediction):
		"""
		Peform binary classification on the class probabilities.

		Args:
		-----
			prediction: An N x k array of class probabilities.

		Returns:
		--------
			An N x k array of binary class assignments.
		"""
		N, k = prediction.shape
		clasfn = np.zeros((N, k))

		for row in xrange(N):
			ind = np.where(prediction[row] == np.amax(prediction[row]))[0][0]
			#prediction[row, :] = np.zeros(k)
			clasfn[row, ind] = 1

		return clasfn


def testMnist():
	"""
	Test cnn using the MNIST dataset.
	"""

	print "Loading MNIST images..."
	data = np.load('data/cnnMnist.npz')
	train_data = data['train_data'][0:10000].reshape(10000, 28, 28, 1)
	valid_data = data['valid_data'][0:1000].reshape(1000, 28, 28, 1)
	test_data = data['test_data'].reshape(10000, 28, 28, 1)
	train_label = data['train_label'][0:10000]
	valid_label = data['valid_label'][0:1000]
	test_label = data['test_label']

	print "Initializing network..."
	layers = {
		"fully-connected": [PerceptronLayer(10, 150, "softmax"), PerceptronLayer(150, 256, "tanh")],
		# Ensure size of output maps in preceeding layer is equals to the size of input maps in next layer.
		"convolutional": [ConvLayer(16, 6, (5,5), (2, 2)), ConvLayer(6, 1, (5,5), (2, 2))]
	}
	cnn = Cnn(layers)
	print "Training network..."
	cnn.train(train_data, train_label, valid_data, valid_label, test_data, test_label, {'learn_rate': 0.1})


def testCifar10():
	"""
	Test the cnn using the CIFAR-10 dataset.
	"""

	print "Loading CIFAR-10 images..."
	data = np.load('data/cifar10.npz')
	train_data = data['train_data'][0:10000]
	valid_data = data['train_data'][10000:11000]
	test_data = data['test_data']
	train_label = data['train_label'][0:10000]
	valid_label = data['train_label'][10000:11000]
	test_label = data['test_label']

	print "Initializing network..."
	layers = {
		"fully-connected": [PerceptronLayer(10, 150, "softmax"), PerceptronLayer(150, 400, "tanh")],
		# Ensure size of output maps in preceeding layer is equals to the size of input maps in next layer.
		"convolutional": [ConvLayer(16, 6, (5,5), (2, 2)), ConvLayer(6, 3, (5,5), (2, 2))]
	}
	cnn = Cnn(layers)
	print "Training network..."
	cnn.train(train_data, train_label, valid_data, valid_label, test_data, test_label, {'learn_rate': 0.1})


if __name__ == '__main__':

	#testMnist()
	testCifar10()
