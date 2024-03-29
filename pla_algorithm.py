# Perceptron Learning Algorithm (PLA) for Binary classification
# Given a training data set where each training sample has a binary label, we want to learn a “line”, which is a hyperplane in a high dimensional space, 
# to separate these training samples into two categories — one for each label. Later, we use the learned line to predict the label of future data elements.
# Please see attached image of algorithm pseudocode from lecture notes


import numpy as np

import sys
sys.path.append("..")

from misc.utils import MyUtils



class PLA:
    def __init__(self):
        self.w = None
        self.degree = 1
        
    def fit(self, X, y, pocket = True, epochs = 100, degree = 1):
        ''' X: n x d matrix 
            y: n x 1 vector of {+1, -1}
            degree: the degree of the Z space
            return w
        '''
        
        self.degree = degree
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)
            
        n, d = X.shape
        X = np.insert(X, 0, 1, axis = 1) # add the column of x_0 = 1 features.
        self.w = np.array([[0],]* (d+1)) # init the w vector

        updated = True
                          
        if not pocket:
            while updated:
                updated = False
                for i in range(n):
                    if np.sign(X[i] @ self.w) != y[i,0]:
                        self.w = self.w + (y[i,0] * X[i]).reshape(-1,1)
                        updated = True
        else:
            errors = n # record the smallest number of errors so far
            best_w = self.w # record the best w vector so far
            while updated and epochs > 0:
                updated = False
                epochs -= 1
                for i in range(n):
                    if np.sign(X[i] @ self.w) != y[i,0]:
                        self.w = self.w + (y[i,0] * X[i]).reshape(-1,1)
                        updated = True
                        cur_errors = self._error_z(X, y)
                        if cur_errors < errors:
                            errors = cur_errors
                            best_w = self.w
            self.w = best_w
                          
        return self.w
    
    def predict(self, X):
        ''' x: n x d matrix 
            return: n x 1 vector, the labels of samples in X
        '''
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)

        X = np.insert(X, 0, 1, axis = 1) # add the x_0 = 1 feature

        return np.sign(X @ self.w)
    

    def _error_z(self, Z, y):
        ''' Used internally by the fit function to count the misclassied samples.
            The sample Z is in the Z space with the bias column.
            
            Z: n x (d'+1) matrix 
            y: n x 1 vector
            
            return: the number of misclassifed elements in Z using self.w
        '''
                    
        n = Z.shape[0]

        # this is better code than the loop below but needs a test when time is available        
        y_hat = Z @ self.w
        y_hat = np.sign(y_hat)
        errors = n - np.sum(y_hat == y)
                
        return errors


    def error(self, X, y):
        ''' X: n x d matrix 
            y: n x 1 vector
            return the number of misclassifed elements in X using self.w
        '''
        
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)
            
        X = np.insert(X, 0, 1, axis = 1) # add the column of x_0 = 1 features.

        # this is better code than the loop below but needs a test when time is available        
        y_hat = X @ self.w
        y_hat = np.sign(y_hat)
        errors = np.sum(y_hat != y)
                
        return errors
