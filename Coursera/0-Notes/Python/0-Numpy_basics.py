"""------------------------------------------------ Numpy_basics.py ------
    |   Purpose: First examples of Python programming using the numpy
    |       package.
    |
    |   Developer:  
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------"""

#!/usr/bin/env python3
import math
import numpy as np

def sigmoid(x):
    """
      Given the nature of deep learning, numpy is commonly used due to its
      matrix and vector input options
    """
    s = 1/(1+np.exp(-x)) 
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = (s*(1-s))
    return ds

def image2vector(image):
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]),1)
    return v

def normalizeRows(x):
    x_normalized = ( np.linalg.norm(x, axis = 1, keepdims = True) )
    x = x / x_normalized 
    return x

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp/x_sum
    return s

def L1 (yhat, y):
    loss = np.sum(abs(y - yhat), axis = 0)
    return loss

def L2 (yhat, y):
    loss = np.sum(np.dot((y - yhat), (y - yhat)), axis = 0)
    return loss

def main():
    x = np.array([1, 2, 3])
    print("Sigmoid: "+ str(sigmoid(x)))
    print("Sigmoid derivarive: " + str(sigmoid_derivative(x)))
    
    image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

    print("Test of reshaped image: \n" + str(image2vector(image)))

    x = np.array([
        [0, 3, 4],
        [1, 6, 4]])
    print("Normalized row example:\n" + str(normalizeRows(x)))

    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0 ,0]])
    print("Softmax implementation:\n" + str(softmax(x)))

    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1:" + str(L1(yhat,y)))

    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L2 = " + str(L2(yhat,y)))

if __name__ == '__main__':
    main()
