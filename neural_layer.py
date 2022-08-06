# Implementation of one layer used in the forward feeding and back propagation neural network 

import numpy as np
import code_NN.math_util as mu



class NeuralLayer:
    def __init__(self, d = 1, act = 'tanh'):
        ''' d: the number of NON-bias nodes in the layer
                             
            act: the activation function. It will not be useful/used, regardlessly, at the input layer.
                 1) 'tanh': the tanh function
                 2) 'logis': the logistic function
                 3) 'iden': the identity function
                 4) 'relu': the ReLU function 
        '''

        self.d = d   # the number of non-bias nodes

        self.act = eval('mu.MyMath.' + act)   # the activation function, not useful/used at the input layer
        self.act_de = eval('mu.MyMath.' + act + '_de')  # the derivative of the activation function, not useful/used at the input layer 
        
        # The following matrix/vectors are to be materalized by the NN-level code. Some are not useful for the input layer. 
        # Below, N' represents the minibatch size, \ell represents the index of this layer. 
        self.S = None       # N' x d^{(\ell)} matrix. Each row is the vector of the d signals, sent into the d nodes, by each sample. Not useful for the input layer. 
        self.X = None       # N' x (d^{(\ell)}+1) matrix. Each row is the vector of the d+1 outputs, sent out by the bias node and the d neurons, by each sample.
        self.Delta = None   # N' x d^{(\ell)} matrix. Each row is vector of delta = \partial E / \partial S, where E is the error. Not useful for the input layer
        self.G = None       # (d^{(\ell-1)}+1 ) x d^{(\ell)} matrix. The gradient of E over W.
        self.W = None       # (d^{(\ell-1)}+1 ) x d^{(\ell)} matrix. The weights of the edges coming into layer \ell.
