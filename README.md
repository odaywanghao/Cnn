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

To use the cnn, create a dictionary containing a list of fully-connected layers and convolutional layers.

`{`
`	"fully-connected": [],`
`	"convolutional"  : []`
`}`

A convolutional layer is created by specifying the number of kernels in the layer, size of each kernel and the sampling factor as tuples.

`ConvLayer(16, (5,5), (2, 2))`

Likewise a layer in the fully connected layer is created by specifying the number of incoming units into the layer, number of outgoing units from the layer and the type of activation function for the outgoing layers.

`PerceptronLayer(10, 150, "softmax")`

Currently 3 types of activation functions are supported in the perceptron layer:

`
* Softmax
* Hyperbolic tangent
* Sigmoid
`

While the convolutional layer only supports the Hyperbolic tanget activation function.

In each list, the layers should should be ordered heirarchically whereby the topmost layer is at the beginning of the list and so on.

`
{
	"fully-connected": [],
	"convolutional"  : [ ConvLayer(16, (5,5), (2, 2)),
					     ConvLayer( 6, (5,5), (2, 2))
					   ]
}`

Ensure that the number of incoming units to the fully connected layer is equal to the total number of downsampled units in the last convolutional layer. Next release will automatically correct inconsistencies between both layers. 

`
Input image size: 28 x 28,

{
	"fully-connected": [ ...,
						 PerceptronLayer(150, 256, "tanh")
					   ],
	"convolutional"  : [ ConvLayer(16, (5,5), (2, 2)),
					     ConvLayer( 6, (5,5), (2, 2))
					   ]
}`

Initialize the network using the dictionary of layers.

`
Cnn({
		"fully-connected": [ PerceptronLayer( 10, 150, "softmax"),
							 PerceptronLayer(150, 256, "tanh")
						   ],
		"convolutional"  : [ ConvLayer(16, (5,5), (2, 2)),
						     ConvLayer( 6, (5,5), (2, 2))
						   ]
 	})`


### Training and Prediction

Training and prediction on the network are done by calling the `train()` and `predict()` functions respectively.

**Note** The `predict()` function returns a probability distributions over the various classes. You can however call the `classify()` function on its output to get the most probable class for each data instance.

To Do
-----

1. Check and automatically rectify inconsistent connection between fully-connected and convolutional layers of network.
2. Allow exportation and importation of network weights.
3. Implement drop-out.
4. Add tool for visualization of kernels.
5. Allow GPU utilization.
6. Implement techniques for smoother learning: momentum and weight penalties.

References
----------

* le Cun (1989) *Generalization and Network Design Strategies*. 
* le Cun and Bengio (1995) *Convolutional Networks for Images, Speech and Time-Series*.
* Fukushima (1980) *Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position*.
* RumelHart, Hinton & Williams (1986) *Learning Representations by Back-Propagating Errors*.
* Bouvrie (2006) *Notes on Convolutional Neural Networks*.
 