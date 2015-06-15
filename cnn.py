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
import matplotlib.pyplot as plt
import pickle as pkl
from theano.tensor.signal.downsample import max_pool_2d as max_pool
from theano.tensor.signal.downsample import max_pool_2d_same_size as max_pool_same
from skimage.transform import downscale_local_mean as downsample
from copy import deepcopy
from mlp import *
from util import *


def centerDataset(data):
	return data


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


class PoolLayer():
	"""
	A pooling layer.
	"""

	def __init__(self, factor, poolType='avg'):
		"""
		Initialize the pooling layer.

		Args:
		-----
			factor: Tuple repr. pooling factor.
			poolType: String repr. the pooling type i.e 'avg' or 'max'.
		"""
		self.type, self.factor, self.grad = poolType, factor, None


	def bprop(self, dEdo):
		"""
		Compute error gradients and return sum of error from output down 
		to this layer.

		Args:
		-----
			dEdo: A N x k x m2 x n2 array of errors from prev layers.

		Returns:
		--------
			A N x k x x m1 x m1 array of errors.
		"""
		if self.type == 'max':
			return np.kron(dEdo, np.ones(self.factor)) * self.grad
		else:
			return np.kron(dEdo, np.ones(self.factor)) * (1.0 / np.prod(self.factor))
			

	def update(self, eps_w, eps_b, mu, l2):
		"""
		Update the weights in this layer.

		Args:
		-----
			eps: Learning rate.
		"""
		pass #Nothing to do here :P


	def feedf(self, data):
		"""
		Pool features within a given receptive from the input data.

		Args:
		-----
			data: An N x k x m1 x n1 array of input plains.

		Returns:
		-------
			A N x k x m2 x n2 array of output plains.
		"""
		if self.type == 'max':
			x = tn.dtensor4('x')
			f = thn.function([x], max_pool(x, self.factor))
			g = thn.function([x], max_pool_same(x, self.factor))
			self.grad = g(data + 0.0000000001) / (data + 0.0000000001) #Pick up max inactive units.
			self.grad[np.where(np.isnan(self.grad))] = 0
			return f(data)
		else:
			return downsample(data, (1, 1, self.factor[0], self.factor[1]))


class ConvLayer():
	"""
	A Convolutional layer.
	"""

	def __init__(self, k, l, ksize, outputType='relu', init_w=0.01, init_b=0):
		"""
		Initialize convolutional layer.

		Args:
		-----
			k: No. feature maps in layer.
			l: No. input planes in layer or no. channels in input image.
			ksize: Tuple repr. the size of a kernel.
			outputType: String repr. type of non-linear activation i.e 'relu', 'tanh' or 'sigmoid'.
			init_w: Std dev of initial weights drawn from a std Normal distro.
			init_b: Initial value of biases.
		"""
		self.o_type = outputType
		self.kernels = init_w * np.random.randn(k, l, ksize[0], ksize[1])
		self.bias = init_b * np.ones((k, 1, 1))
		self.v_w, self.v_b = 0, 0


	def bprop(self, dEdo):
		"""
		Compute error gradients and return sum of error from output down
		to this layer.

		Args:
		-----
			dEdo: A N x k x m2 x n2 array of errors from prev layers.

		Returns:
		-------
			A N x l x m1 x n1 array of errors.
		"""
		if self.o_type == 'sigmoid':
			dEds = dEdo * sigmoid(self.maps + self.bias) * (1 - sigmoid(self.maps + self.bias))
		elif self.o_type == 'tanh':
			dEds = dEdo * sech2(self.maps + self.bias)
		else:
			dEds = dEdo * np.where((self.maps + self.bias) > 0, 1, 0)

		self.dEdb = np.sum(np.sum(np.average(dEds, axis=0), axis=1), axis=1).reshape(self.bias.shape)

		#Correlate.
		xs, dEds = np.swapaxes(self.x, 0, 1), np.swapaxes(dEds, 0, 1)
		self.dEdw = fastConv2d(xs, rot2d90(dEds, 2)) / dEdo.shape[0]
		self.dEdw = np.swapaxes(self.dEdw, 0, 1)
		self.dEdw = rot2d90(self.dEdw, 2)

		#Correlate
		dEds, kernels = np.swapaxes(dEds, 0, 1), np.swapaxes(self.kernels, 0, 1)
		return fastConv2d(dEds, rot2d90(kernels, 2), 'full')


	def update(self, eps_w, eps_b, mu, l2):
		"""
		Update the weights in this layer.

		Args:
		-----
			eps_w, eps_b: Learning rates for the weights and biases.
			mu: Momentum coefficient.
			l2: L2 Regularization coefficient.
		"""
		self.v_w = (mu * self.v_w) - (eps_w * self.dEdw) - (eps_w * l2 * self.kernels)
		self.v_b = (mu * self.v_b) - (eps_b * self.dEdb) - (eps_b * l2 * self.bias)
		self.kernels = self.kernels + self.v_w
		self.bias = self.bias + self.v_b


	def feedf(self, data):
		"""
		Return the non-linear result of convolving the input data with the
		weights in this layer.

		Args:
		-----
			data: An N x l x m1 x n1 array of input plains.

		Returns:
		-------
			A N x k x m2 x n2 array of output plains.
		"""
		self.x = data
		self.maps = fastConv2d(self.x, self.kernels)

		if self.o_type == 'tanh':
			return np.tanh(self.maps + self.bias)
		elif self.o_type == 'sigmoid':
			return sigmoid(self.maps + self.bias)

		return relu(self.maps + self.bias)


class Cnn():
	"""
	Convolutional neural network class.
	"""

	def __init__(self, layers={}):
		"""
		Initialize network.

		Args:
		-----
			layers: Dict. of fully connected and convolutional layers arranged heirarchically.
		"""
		self.layers, self.div_x_shape, self.div_ind = [], None, 0
		if layers != {}:
			self.layers = deepcopy(layers["fc"]) + deepcopy(layers["conv"])
			self.div_x_shape = None
			self.div_ind = len(layers["fc"])


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

		#Notify fc of training.
		for layer in self.layers[0 : self.div_ind]:
			layer.train = True

		#Check the parameters

		print "Training network..."
		plt.ion()
		N, itrs, train_errors, valid_errors = train_data.shape[0], 0, [], []

		for epoch in xrange(params['epochs']):

			avg_train_errors, avg_valid_errors = [], []

			start, stop = range(0, N, params['batch_size']), range(params['batch_size'], N, params['batch_size'])

			for i, j in zip(start, stop):

				pred = self.predict(train_data[i:j])
				self.backprop(pred - train_label[i:j])
				self.update(params, itrs)

				tc, vc  = self.classify(pred), self.classify(self.predict(valid_data))
				ce_train, ce_valid = mce(tc, train_label[i:j]), mce(vc, valid_label)

				print '\r| Epoch: {:5d}  |  Iteration: {:8d}  |  Train mce: {:.2f}  |  Valid mce: {:.2f} |'.format(epoch, itrs, ce_train, ce_valid)
				if epoch != 0 and epoch % 100 == 0:
  					print '--------------------------------------------------------------------------------'
  				if params['view_kernels']:
  					self.displayKernels()

  				itrs = itrs + 1
  				avg_train_errors.append(ce_train)
  				avg_valid_errors.append(ce_valid)

  			i = start[-1]
  			pred = self.predict(train_data[i:])
			self.backprop(pred - train_label[i:])
			self.update(params, itrs)

			tc, vc  = self.classify(pred), self.classify(self.predict(valid_data))
			ce_train, ce_valid = mce(tc, train_label[i:]), mce(vc, valid_label)

			print '\r| Epoch: {:5d}  |  Iteration: {:8d}  |  Train mce: {:.2f}  |  Valid mce: {:.2f} |'.format(epoch, itrs, ce_train, ce_valid)
			if epoch != 0 and epoch % 100 == 0:
  				print '--------------------------------------------------------------------------------'
  			if params['view_kernels']:
  				self.displayKernels()

  			itrs = itrs + 1
  			avg_train_errors.append(ce_train)
  			avg_valid_errors.append(ce_valid)

  			plt.figure(2)
  			plt.show()
  			train_errors.append(np.average(avg_train_errors))
  			valid_errors.append(np.average(avg_valid_errors))
  			plt.xlabel('Epochs')
  			plt.ylabel('Train (green) and Valid (blue) mce')
  			plt.plot(range(epoch + 1), train_errors, '-g')
  			plt.plot(range(epoch + 1), valid_errors, '-b')
  			plt.axis([0, params['epochs'], 0, 1.00])
  			plt.draw() #Add fucking legend later.

  		#Stop fc dropout after training.
		for layer in self.layers[0 : self.div_ind]:
			layer.train = False
		plt.ioff()

  		tc = self.classify(self.predict(test_data))
  		print '\nTest mce: {:.2f}'.format(mce(tc, test_label))

  		#Also add training intervals to save architecture.
  		print "Saving model..."
  		self.saveModel('mnist_cnn_1')

  		print "Done."


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


	def update(self, params, i):
		"""
		Update the network weights.

		Args:
		-----
			params: Training parameters.
		"""
		fc, conv = params['fc'], params['conv']

		eps_w = epsilon_decay(fc['eps_w'], fc['eps_decay'], fc['eps_satr'], i, fc['eps_intvl'])
		eps_b = epsilon_decay(fc['eps_b'], fc['eps_decay'], fc['eps_satr'], i, fc['eps_intvl'])

		for layer in self.layers[0 : self.div_ind]:
			layer.update(eps_w, eps_b, fc['mu'], fc['l2'])

		eps_w = epsilon_decay(conv['eps_w'], conv['eps_decay'], conv['eps_satr'], i, conv['eps_intvl'])
		eps_b = epsilon_decay(conv['eps_b'], conv['eps_decay'], conv['eps_satr'], i, conv['eps_intvl'])

		for layer in self.layers[self.div_ind:]:
			layer.update(eps_w, eps_b, conv['mu'], conv['l2'])


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


	def saveModel(self, filename):
		"""
		Save the current network model in file filename.

		Args:
		-----
			filename: String repr. name of file.
		"""
		model = {
			'fc': self.layers[0 : self.div_ind],
			'conv': self.layers[self.div_ind :]
		}

		f = open(filename, 'w')
		pkl.dump(model, f, 1)
		f.close()


	def loadModel(self, filename):
		"""
		Load an empty architecture with the network model
		saved in file filename.

		Args:
		-----
			filename: String repr. name of file.
		"""
		f = open(filename, 'r')
		model = pkl.load(f)

		if model != {} and self.layers == []:
			self.layers = model["fc"] + model["conv"]
			self.div_x_shape = None
			self.div_ind = len(model["fc"])

		f.close()


	def displayKernels(self):
		"""
		Displays the kernels in the first convolutional layer.
		"""
		plt.figure(1)

		kernels = self.layers[len(self.layers) - 1].kernels
		k, l, m, n = kernels.shape
		if l == 2 or l > 4:
			print "displayKernels() Error: Invalid number of channels."
			pass

		x = np.ceil(np.sqrt(k))
		y = np.ceil(k/x)

		for i in xrange(k):
			plt.subplot(x, y, i)
			kernel = np.transpose(kernels[i], (1, 2, 0))
			if kernel.shape[2] == 1:
				plt.imshow(kernel[:, :, 0], 'gray')
			else:
				plt.imshow(kernel)
			plt.axis('off')

		plt.draw()


def testMnist():
	"""
	Test cnn using the MNIST dataset.
	"""

	print "Loading MNIST images..."
	data = np.load('data/cnnMnist.npz')
	train_data = data['train_data'][0:1000].reshape(1000, 28, 28, 1)
	valid_data = data['valid_data'][0:1000].reshape(1000, 28, 28, 1)
	test_data = data['test_data'].reshape(10000, 28, 28, 1)
	train_label = data['train_label'][0:1000]
	valid_label = data['valid_label'][0:1000]
	test_label = data['test_label']

	print "Initializing network..."
	# Ensure size of output maps in preceeding layer is equals to the size of input maps in next layer.
	layers = {
		"fc":[
				PerceptronLayer(10, 150, 0.9, 'softmax'),
				PerceptronLayer(150, 256, 0.8, 'tanh')
			],
		"conv":[
				PoolLayer((2, 2), 'max'),
				ConvLayer(16, 6, (5,5)),
				PoolLayer((2, 2), 'max'),
				ConvLayer(6, 1, (5,5))
			]
	}

	params = {
		'epochs': 20,
		'batch_size': 500,
		'view_kernels': True,

		'fc':{
			'eps_w': 0.1,
			'eps_b': 0.1,
			'eps_decay': 0.5,
			'eps_intvl': 0,
			'eps_satr': 'inf',
			'mu': 0.6,
			'l2': 0
		},

		'conv': {
			'eps_w': 0.1,
			'eps_b': 0.1,
			'eps_decay': 0.5,
			'eps_intvl': 0,
			'eps_satr': 'inf',
			'mu': 0.6,
			'l2': 0
		}
	}

	cnn = Cnn(layers)
	cnn.train(train_data, train_label, valid_data, valid_label, test_data, test_label, params)


def testCifar10():
	"""
	Test the cnn using the CIFAR-10 dataset.
	"""

	print "Loading CIFAR-10 images..."
	data = np.load('data/cifar10.npz')
	train_data = data['train_data'][0:49500]
	valid_data = data['train_data'][49500:50000]
	test_data = data['test_data']
	train_label = data['train_label'][0:49500]
	valid_label = data['train_label'][49500:50000]
	test_label = data['test_label']

	#print "Centering images..."

	print "Initializing network..."
	# Ensure size of output maps in preceeding layer is equals to the size of input maps in next layer.
	layers = {
		"fc":[
				PerceptronLayer(10, 64, 0.9, 'softmax'),
				PerceptronLayer(64, 64, 0.8, 'tanh')
			],
		"conv":[
				ConvLayer(64, 32, (5,5)),
				PoolLayer((2, 2), 'avg'),
				ConvLayer(32, 32, (5,5)),
				PoolLayer((2, 2), 'max'),
				ConvLayer(32, 3, (5,5), init_w=0.0001)
			]
	}

	params = {
		'epochs': 30,
		'batch_size': 128,
		'view_kernels': True,

		'fc':{
			'eps_w': 0.001,
			'eps_b': 0.002,
			'eps_decay': 9,
			'eps_intvl': 0,
			'eps_satr': 'inf',
			'mu': 0.6,
			'l2': 0.03
		},

		'conv': {
			'eps_w': 0.001,
			'eps_b': 0.002,
			'eps_decay': 9,
			'eps_intvl': 0,
			'eps_satr': 'inf',
			'mu': 0.6,
			'l2': 0.004
		}
	}

	cnn = Cnn(layers)
	cnn.train(train_data, train_label, valid_data, valid_label, test_data, test_label, params)



if __name__ == '__main__':

	testMnist()
	#testCifar10()