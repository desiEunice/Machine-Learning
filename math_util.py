import numpy as np



# Various math functions, including a collection of activation functions used in NN.




class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        
        return np.tanh(x)

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        tanhde = 1 - (np.tanh(x)**2)
        return tanhde
        

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''
        #x = np.array(x)
        def logis_f(x):
            return (1 / (1 + np.exp(-x)))

        logis_v = np.vectorize(logis_f)
        return logis_v(x)

        
         
        
        
        

    

    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        x = np.array(x)
        #log_de = (1/(1+np.exp(-x)))* (np.exp(-x)/(1+np.exp(-x)))
        
        log_de = MyMath.logis(x) * (1 - MyMath.logis(x))
        
        return log_de


    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        x = np.array(x)
        
        return x


    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        
        x = np.array(x)
        
        return np.ones(x.shape)
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        def relu_f(x): 
            return np.maximum(0,x)
        
        relu_v =  np.vectorize(relu_f)
        return relu_v(x)
       

    
    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.
        
            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        x = np.array(x)
        x[x<0] = 0
        x[x>0] = 1 

        return x
        #return np.where(x > 0, 1, 0)

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.   
        '''
        relu_d = np.vectorize(MyMath._relu_de_scaler)
        return relu_d(x)
    

    
