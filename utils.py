# Implementation of functions for data manipulation for machine learning programs. 
# Author: Eunice Ofori-Addo


# Various tools for data manipulation. 
# Z transformation function. Support the pocket version for linearly non separable data. 




import numpy as np
import math


class MyUtils:

    
    def z_transform(X, degree = 2):
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
        
        
        
        r = degree
        if r == 1:
            return X

        Z = X.copy()
        
        #r = degree
        n,d = X.shape
        B = []
        
        for i in range(r):
            B.append(math.comb(d+i, d-1))
            
        l = np.arange(np.sum(B))
            
        q = 0
        p = d
        g = d
        
      
        
        for i in range(1,r):
            for j in range(q, p):
                
                head = l[j]
                for k in range(head, d):
                    temp = (Z[:,j] * X[:,k]).reshape(-1,1)
                    Z = np.append(Z, temp, axis = 1)
                    l[g] = k
                    g = g + 1
                    
            q = p 
            p = p + B[i]
                
        return Z
        
        
        

