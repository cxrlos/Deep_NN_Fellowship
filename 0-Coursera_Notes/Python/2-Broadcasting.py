"""------------------------------------------------ Broadcasting.py ------
    |   Purpose: First approach on broadcasting.
    |
    |   Developer:  
    |       Carlos García - https://github.com/cxrlos
    |
    *------------------------------------------------------------------"""

import numpy as np
A = np.array([
    [56.0, 0.0, 4.4, 68.0], 
    [1.2, 104.0, 52.0, 8.0], 
    [1.8, 135.0, 99.0, 0.9]])

print(A)

# Axis = 0 sums vertically and axis = 1 horizontally
calories = A.sum(axis = 0)
print(calories)

# Broadcating, (3,4) matrix divided by (1,4) matrix
# In this case, calories is already (1,4)
percentage = 100 * A/calories.reshape(1,4)
print(percentage)


