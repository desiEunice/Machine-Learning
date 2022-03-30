# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent
# Author: Eunice Ofori-Addo


import numpy as np
import math
import sklearn
from sklearn.datasets import make_blobs
import sys
sys.path.append("..")

from code_misc.utils import MyUtils



class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''

        self.degree = degree
        X = MyUtils.z_transform(X, degree = self.degree)

        if SGD:
            self._fit_SGD(X, y, lam, eta, iterations, mini_batch_size)    ## SGD - stochastic gradient descent
        else:
            self._fit_BGD(X, y, lam, eta, iterations)    ## BGD - batch gradient descent


        
    def _fit_BGD(self, X, y, lam = 0, eta = 0.01, iterations = 1000):    

        X = np.insert(X, 0, 1, axis = 1)   # add bias column
        n, d = X.shape
        self.w = np.zeros(d).reshape(-1,1)

        

        for iter in range(iterations):

            ## Without regularization
            # self.w = self.w + (eta/n)*(X.T.dot(y * LogisticRegression._v_sigmoid(-s))) # add class name._v_sigmoid (since _v.sigmoid is not a defined object)

            ## With regularization
            s = y * (X.dot(self.w))
            self.w = (1 - (2*lam*eta)/n) * self.w + (eta/n)*(X.T.dot(y * LogisticRegression._v_sigmoid(-s)))


    def shuffle_data(self,X,Y):
        shuffle = np.random.permutation(X.shape[0])
        X_shuffle = X[shuffle]
        Y_shuffle = Y[shuffle]

        return X_shuffle, Y_shuffle



    def _fit_SGD(self, X, Y, lam = 0, eta = 0.01, iterations = 1000, mini_batch_size = 1):
        
        X = np.insert(X, 0, 1, axis = 1)
        n, d = X.shape
        self.w = np.zeros(d).reshape(-1,1)


        mini_batches = [] 
        no_of_batches = math.ceil(n/mini_batch_size)
        #X,Y = self.shuffle_data(X,y)



        for i in range(no_of_batches):    ## creating mini_batches for SGD
            i_mini = (i * mini_batch_size)  ## starting index
            i1_mini = ((i+1) * mini_batch_size)  ## ending index

            X_mini = X[i_mini : min(i1_mini,n),:]
            Y_mini = Y[i_mini : min(i1_mini,n),:]
            
            mini_batches.append((X_mini, Y_mini))


        for _ in range(iterations):


            for i in mini_batches:

            #for mini_batch in mini_batches
                X_mini, Y_mini = i

                s = Y_mini * (X_mini.dot(self.w))
                self.w = (1 - (2*lam*eta)/n) * self.w + (eta/n)*(X_mini.T.dot(Y_mini * LogisticRegression._v_sigmoid(-s)))



    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis = 1)   # add bias column
        n, d = X.shape

        prob = LogisticRegression._sigmoid(X.dot(self.w))
        return prob 

         
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''

        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis = 1)   # add bias column
        n, d = X.shape

        prob = LogisticRegression._sigmoid(X.dot(self.w))
        pred_labels = np.where(prob > 0.5, +1, -1)

        errors = np.sum(pred_labels != y)
        return errors 



    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
        # Hint: use the np.vectorize API
        vec = np.vectorize(LogisticRegression._sigmoid)
        vsig = vec(s)
        
        return vsig 
   
    
    
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signals
        '''

        sig = 1 / (1 + np.exp(-s))
        return sig           
