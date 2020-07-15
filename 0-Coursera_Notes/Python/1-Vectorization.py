"""------------------------------------------------ Vectorization.py -----
    |   Purpose: First Vectorization approach, compares it to traditional
    |       methods.
    |
    |   Developer:  
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------"""

import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("Vectorized time in ms " + str(1000*(toc-tic)))

c = 0

tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("For looped time in ms " + str(1000*(toc-tic)))
