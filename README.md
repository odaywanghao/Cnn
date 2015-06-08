Convolutional Neural Network
============================

A configurable convolutional neural network for image recognition and classification.


Dependencies
------------

You need the following dependencies to run the network:

* [Numpy] (http://www.numpy.org)
* [Skimage] (http://scikit-image.org)
* [Theano] (http://deeplearning.net/software/theano/)


Usage
-----

### Initialization

To use the cnn, you need to first specify its architecture using a dictionary. The dictionary should contain a list of fully-connected layers and convolutional layers.

	{
		"fully-connected": [],
		"convolutional"  : []
	}

A convolutional layer is created by specifying the number of kernels in the layer, size of each kernel and the sampling factor as tuples.

	ConvLayer(16, (5,5), (2, 2))

Likewise a fully-connected layer is created by specifying the number of outgoing units from the layer, number of incoming units to the layer and the type of activation function for the layer.

	PerceptronLayer(10, 150, "softmax")

Currently 3 types of activation functions are supported by the perceptron layer:

* Softmax
* Hyperbolic tangent
* Sigmoid

The convolutional layer only supports the Hyperbolic tanget activation function.

In each list, the layers should be ordered heirarchically whereby the topmost layer is at the beginning of the list and so on.


	{
		"fully-connected": [],
		"convolutional"  : [ ConvLayer(16, (5,5), (2, 2)),
						     ConvLayer( 6, (5,5), (2, 2))
						   ]
	}

Ensure that the number of incoming units to the first fully-connected layer is equal to the total number of downsampled units in the last convolutional layer. Future versions of the network will automatically correct inconsistencies between both layers at initialization. 

	Input image size: 28 x 28

	{
		"fully-connected": [ ...,
							 PerceptronLayer(150, 256, "tanh")
						   ],
		"convolutional"  : [ ConvLayer(16, (5,5), (2, 2)),
						     ConvLayer( 6, (5,5), (2, 2))
						   ]
	}

Finally, initialize the network with the specified architecture.

	Cnn({
			"fully-connected": [ PerceptronLayer( 10, 150, "softmax"),
								 PerceptronLayer(150, 256, "tanh")
							   ],
			"convolutional"  : [ ConvLayer(16, (5,5), (2, 2)),
							     ConvLayer( 6, (5,5), (2, 2))
							   ]
	 	})


### Training and Prediction

Training and prediction on the network are done by calling the `train()` and `predict()` functions respectively.

**Note** The `predict()` function returns a probability distributions over the various classes. You can however call the `classify()` function on its output to get the most probable class for each data instance.

To Do
-----

1. Check and automatically rectify inconsistent connection between fully-connected and convolutional layers of network.
2. Allow exportation and importation of network weights.
3. Add tool for visualization of kernels.
4. Allow GPU utilization.

References
----------

* le Cun (1989) *Generalization and Network Design Strategies*. 
* le Cun & Bengio (1995) *Convolutional Networks for Images, Speech and Time-Series*.
* Fukushima (1980) *Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position*.
* Rumelhart, Hinton & Williams (1986) *Learning Representations by Back-Propagating Errors*.
* Bouvrie (2006) *Notes on Convolutional Neural Networks*.
* Andrade (2014) *Best Practices for Convolutional Neural Networks Applied to Object Recognition in Images*.
* Sutskever, Martens, Dahl & Hinton (2013) *On the importance of initialization and momentum in deep learning*.
* Krizhevsky, Sutskever & Hinton (2012) *ImageNet Classification with Deep Convolutional Neural Networks*.
* Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014) *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
 