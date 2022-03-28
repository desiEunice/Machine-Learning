
# Implementation of the linear regression with L2 regularization.
# It supports the closed-form method and the gradient-desecent based method. 
# Author: Eunice Ofori-Addo



import numpy as np
import math
import sys
sys.path.append("..")



class LinearRegression:
    def __init__(self):
        self.w = None   # The (d+1) x 1 numpy array weight matrix
        self.degree = 1
        self.MSE = None
        
        
    def fit(self, X, y, CF = True, lam = 0, eta = 0.01, epochs = 1000, degree = 1):
        ''' Find the fitting weight vector and save it in self.w. 
            
            parameters: 
                X: n x d matrix of samples, n samples, each has d features, excluding the bias feature
                y: n x 1 matrix of lables
                CF: True - use the closed-form method. False - use the gradient descent based method
                lam: the ridge regression parameter for regularization
                eta: the learning rate used in gradient descent
                epochs: the maximum epochs used in gradient descent
                degree: the degree of the Z-space
        '''
        self.degree = degree
        X = MyUtils.z_transform(X, degree = self.degree)
        
        if CF:
            self._fit_cf(X, y, lam)
        else: 
            self._fit_gd(X, y, lam, eta, epochs)
 


            
    def _fit_cf(self, X, y, lam = 0):
        ''' Compute the weight vector using the clsoed-form method.
            Save the result in self.w
        
            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        ''' 

        
        X = np.insert(X, 0, 1, axis = 1)   # add bias column
        n, d = X.shape


        XTX = X.T.dot(X) 
        XTy = X.T.dot(y) 
        I = np.eye(d)

        self.w = np.linalg.pinv(XTX + (lam*I)) @ (XTy)        
       
    
    
    def _fit_gd(self, X, y, lam = 0, eta = 0.01, epochs = 1000):
        ''' Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''


        X = np.insert(X, 0, 1, axis = 1)   # add bias column
        n, d = X.shape

        self.w = np.zeros(d).reshape(-1,1)

        XTX = X.T.dot(X)  
        XTy = X.T.dot(y)
        I = np.eye(d)

        xx = I - ((2*eta)/n) * (XTX + (lam*I))
        xy = ((2*eta)/n) * XTy      
 

        for epoch in range(epochs):
            self.w = (xx @ self.w ) + xy             



    
    def predict(self, X):
        ''' parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        '''

        
        X = MyUtils.z_transform(X, degree = self.degree) # add z transsformation 
        X = np.insert(X, 0, 1, axis = 1) # add bias column to x
       
        y_pred = X.dot(self.w)
        return y_pred


    
    def error(self, X, y, epochs = 1000):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''


        X = MyUtils.z_transform(X, degree = self.degree)
        X = np.insert(X, 0, 1, axis = 1) # add bias column to x


        n, d = X.shape        

        
        self.MSE = np.square((X @self.w) - y).mean()
        return self.MSE
        


