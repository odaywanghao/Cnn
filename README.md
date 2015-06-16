Convolutional Neural Network
============================

A non-optimized convolutional neural network (CNN) for experimentation on image recognition and classification. NOT RECOMMENDED FOR USE ON VERY LARGE DATASETS. For such you may want to use [cuda-convnet](https://code.google.com/p/cuda-convnet/), however u're free to tinker with this :D


Dependencies
------------

You need the following dependencies to run the network:

* [Numpy] (http://www.numpy.org)
* [Skimage] (http://scikit-image.org)
* [Theano (Bleeding-edge)] (http://deeplearning.net/software/theano/install.html#install-bleeding-edge)


Initialization
--------------

To initialize the cnn, you need to first specify its architecture. Here the architecture consists of two "levels". The first, `fc`, contains multiple perceptron layers a.k.a fully-connected layers while the second, `conv`, contains a combination of Pooling and Convolutional layers.

These two levels are defined in a dictionary as ffs.

	{
		"fc": [],
		"conv"  : []
	}

All layers are arranged heirarchically, with the topmost layer in each level placed at the beginning of the list. Once this is done, the network can be created by passing the dictionary. Optionally an empty dictionary can be used to initialize the network if an exsisting network model is to be used by calling the CNN's `loadModel()` method.

It is important to note that when ordering the layers, the number of outgoing units from the topmost `conv` layer MUST equal the number of incoming units to the bottommost `fc` layer. Hence some "calculations" will need to be done across all layers to ensure that each layer is ready to recieve the right number of units from the preceeding layer. This is further explained below.


### Fully-Connected Layer

A fully-connected layer is created using the `PerceptronLayer class

	PerceptronLayer(no_outputs, no_inputs, prob=1, outputType='relu', init_w=0.01, init_b=0)

This layer supports 5 types of activation functions:

* Rectifier
* Hyperbolic tangent
* Softmax
* Sigmoid
* Linear

The probability of a neuron being present during dropout is specified through the `prob` parameter.


### Convolutional Layer

A convolutional layer is created using the `ConvLayer` class. *briefly state what it does*

	ConvLayer(noKernels, channels, kernelSize, outputType='relu', stride=1, init_w=0.01, init_b=0)

This layer supports 3 types of activation functions:

* Rectifier
* Hyperbolic tangent
* Sigmoid

*Talk about stride, input and how output of this layer is calculated.*


### Pooling Layer

This layer typically follows the convolutional layer and *briefly state what it does*

	PoolLayer(factor, poolType='avg')

This layer supports 2 the two common types of pooling functions:

* Max pooling
* Mean Pooling/subsampling

*Talk about input & how output of this layer is calculated.*


Training
--------

*Talk about parameter file.*
*Talk about the various training parameters.*

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
		"fc":[
				PerceptronLayer(10, 64, outputType='softmax'),
				PerceptronLayer(64, 64)
			],

		"conv":[
				PoolLayer((2, 2)),
				ConvLayer(32, 3, (5,5))
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

To use dropout, the probability of a neuron being present needs to be specified and just before training, the layer's 'train' property set to True. Once training ends, this should be set back to False. 

**Note** The `predict()` function returns a probability distributions over the various classes. You can however call the `classify()` function on its output to get the most probable class for each data instance.

To Do
-----

1. Improve exportation and importation of network model.
2. Implement GPU utilization.
3. Implement local contrast normalization.

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