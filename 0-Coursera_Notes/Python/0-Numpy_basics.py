"*------------------------------------------------- Numpy_basics.py ------
    |   Purpose: First examples of Python programming using the numpy
    |       package.
    |
    |   Developer:  
    |       Carlos García - https://github.com/cxrlos
    |
    *-------------------------------------------------------------------*"

#!/usr/bin/env python3
import math
import numpy as np

# Using math insted of numpy
def basic_sigmoid(x):
    s = 1/(1+math.exp(-x))
    return s

# Using numpy
def sigmoid(x):
    s = None
    " 
      Given the nature of deep learning, numpy is commonly used due to its
      matrix and vector input options
      x = np.array([1, 2, 3])
      print (x + 3)
      print(np.exp(x))
      = 1/(1+np.exp(-x))
    "
    return s

def sigmoid_derivative(x):
    s = 1/(1+np.exp(-x))
    ds = (s*(1-s))
    return ds

def main():
    res = basic_sigmoid(3)
    print(res)
    x = np.array([1, 2, 3])
    res = sigmoid(x)
    print(res)
    x = np.array([1, 2, 3])
    print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

if __name__ == '__main__':
    main()
