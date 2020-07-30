import numpy as np
import matplotlib.pyplot as plt

# 0이상에서는 linear, 0이하에서는 0
def relu(x):
    return np.maximum(0,x)

x = np.arange(-5,5,0.1)
y = relu(x)


plt.plot(x,y)
plt.grid()
plt.show()

"""
leakyrelu, elu, selu
0이하에서 수렴한다.
"""