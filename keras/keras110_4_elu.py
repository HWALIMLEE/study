import numpy as np
import matplotlib.pyplot as plt

# relu--->leakyrelu--->elu--->selu

# def elu(x):
#       x = np.copy(x)
#       x[x<0]=0.2*(np.exp(x[x<0])-1)
#       return x

# x = np.arange(-5,5,0.1)
# y = elu(x)

# plt.plot(x,y)
# plt.grid()
# plt.show()


def elu(x):
    y_list = []
    for x in x:
        if(x>0):
            y = x
        if(x<0):
            y = 0.2*(np.exp(x)-1)
        y_list.append(y)
    return y_list

x = np.arange(-5,5,0.1)
y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()

# a = 0.2
# x = np.arange(-5,5,0.1)
# y = [x if x>0 else a*(np.exp(x)-1) for x in x]