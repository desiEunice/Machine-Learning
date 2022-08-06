# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



from math import sqrt
import numpy as np
import math


from code_NN.nn_layer import NeuralLayer
import code_NN.math_util as mu
import code_NN.nn_layer

import random


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
                     
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        self.L += 1
        self.layers.append(NeuralLayer(d,act))

    
    
    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''

        weight_rng = np.random.default_rng(2142)
        for l in range(1, self.L+1):  
            self.layers[l].W = weight_rng.uniform(-1/sqrt(self.layers[l].d), 1/sqrt(self.layers[l].d), (self.layers[l-1].d +1, self.layers[l].d) )
            #weight_rng.uniform(-1/sqrt(self.layers[l]), 1/sqrt(self.layers[l]), size = (self.layers[l-1] +1, self.layers[l]) )

      
        

    
    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.

        
        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 

        if SGD:
            self._fit_SGD(X, Y, eta, iterations, mini_batch_size)
        else:
            self._fit_BGD(X, Y, eta, iterations)    ## BGD - batch gradient descent

    def shuffle_data(self,X,Y):
        shuffle = np.random.permutation(X.shape[0])
        X_shuffle = X[shuffle]
        Y_shuffle = Y[shuffle]

        return X_shuffle, Y_shuffle


    def _fit_SGD(self, X, Y, eta = 0.01, iterations = 1000, mini_batch_size = 1):

        #np.random.shuffle(X)
        n,d = X.shape

        
        mini_batches = []

        
        X,Y = self.shuffle_data(X,Y)
        #data = np.hstack((X,Y))
        #np.random.shuffle(data)
        

        mini_batches = [] 
        no_of_batches = math.ceil(n/mini_batch_size)
        

        for i in range(no_of_batches):
            i_mini = (i * mini_batch_size)
            i1_mini = ((i+1) * mini_batch_size)

            X_mini = X[i_mini : min(i1_mini,n),:]
            Y_mini = Y[i_mini : min(i1_mini,n),:]
            
            
            mini_batches.append((X_mini, Y_mini))


        for t in range(iterations):
            X,Y = mini_batches[t % no_of_batches]

            self.layers[0].X = np.insert(X, 0, 1, axis = 1)

            for l in range(1, self.L+1):

                current_layer = self.layers[l]
                prev_layer = self.layers[l-1]

                current_layer.S = (prev_layer.X @ current_layer.W)
                S_act = current_layer.act(current_layer.S)
                current_layer.X = np.insert(S_act, 0, 1, axis = 1)


                #Err = np.sum((self.layers[self.L].X[:,1:] - Y)*(self.layers[self.L].X[:,1:] - Y)) * (1/mini_batch_size)
            

                S_de_act = self.layers[self.L].act_de(self.layers[self.L].S)
                self.layers[self.L].Delta = 2 * (self.layers[self.L].X[:, 1:] - Y) * S_de_act
                self.layers[self.L].G = np.einsum('ij,ik -> jk', X, self.layers[self.L-1].X, self.layers[self.L].Delta) * (1/X.shape[0])

                #self.layers.Delta = 2 * (X[:,1] - Y) * NeuralLayer.act_de(self.layers.S[l])
                #self.layers.G = np.einsum('ij,ik -> jk', X, self.layers.Delta) * (1/mini_batch_size)


            for el in range(self.L, 1-1, -1):
                current_layer = self.layers[el]
                prev_layer = self.layers[el-1]
                next_layer = self.layers[el+1]

                S_de_act = current_layer.act_de(next_layer.delta @ next_layer.W[:, 1:].T)
                current_layer = np.einsum('ij,ik -> jk', prev_layer.X, current_layer.Delta) * (1/X.shape[0])



            for ell in range(1, self.L+1):
                current_layer = self.layers[ell]
                current_layer.W = current_layer.W - (eta * current_layer.G)
                #elf.layers.W[ell] = self.layers.W[ell] - (eta * self.layers.G[ell])

                
    def forward_feed(self, X):

        X = np.insert(X, 0, 1, axis = 1)   ## input layer
        

        for l in range(1, self.L + 1):
            current_layer = self.layers[l]
            prev_layer = self.layers[l-1]

            current_layer.S = prev_layer.X @ current_layer.W
            theta_S = current_layer.act(current_layer.S)
            current_layer.X = np.insert(theta_S, 0, 1, axis = 1)

            
        return self.layers[self.L].X[:,1:]



    def _fit_BGD(self, X, Y, eta = 0.01, iterations = 1000):    

         #np.random.shuffle(X)
        n,d = X.shape


        for t in range(iterations):
            
            #X, Y = random.sample(mini_batches, 1) ##how to random sample??

            self.layers[0].X = np.insert(X, 0, 1, axis = 1)

            for l in range(1, self.L+1):

                current_layer = self.layers[l]
                prev_layer = self.layers[l-1]

                current_layer.S = (prev_layer.X @ current_layer.W)
                S_act = current_layer.act(current_layer.S)
                current_layer.X = np.insert(S_act, 0, 1, axis = 1)


                S_de_act = self.layers[self.L].act_de(self.layers[self.L].S)
                self.layers[self.L].Delta = 2 * (self.layers[self.L].X[:, 1:] - Y) * S_de_act
                self.layers[self.L].G = np.einsum('ij,ik -> jk', X, self.layers[self.L-1].X, self.layers[self.L].Delta) * (1/X.shape[0])


            for el in range(self.L, 1-1, -1):
                current_layer = self.layers[el]
                prev_layer = self.layers[el-1]
                next_layer = self.layers[el+1]

                S_de_act = current_layer.act_de(next_layer.delta @ next_layer.W[:, 1:].T)
                current_layer = np.einsum('ij,ik -> jk', prev_layer.X, current_layer.Delta) * (1/X.shape[0])

      
            for ell in range(1, self.L+1):
                current_layer = self.layers[ell]
                current_layer.W = current_layer.W - (eta * current_layer.G)
                #elf.layers.W[ell] = self.layers.W[ell] - (eta * self.layers.G[ell])

        


                
    def forward_feed(self, X):

        X = np.insert(X, 0, 1, axis = 1)   ## input layer
        

        for l in range(1, self.L + 1):
            current_layer = self.layers[l]
            prev_layer = self.layers[l-1]

            current_layer.S = prev_layer.X @ current_layer.W
            theta_S = current_layer.act(current_layer.S)
            current_layer.X = np.insert(theta_S, 0, 1, axis = 1)

            
        return self.layers[self.L].X[:,1:]
    
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''

        return self.forward_feed(X)

    

        
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        n, d = X.shape[0]

        pred = self.predict(X)

        Error = np.sum((pred - Y) * (pred - Y)) * (1/n)

        #X = MyUtils.z_transform(X, degree = self.degree)
        #X = np.insert(X, 0, 1, axis = 1)   # add bias column
   
        #err = np.sum((X[1,:] - Y)*(X[1,:] - Y)) * (1/X.shape[0])

        return Error
        
 
