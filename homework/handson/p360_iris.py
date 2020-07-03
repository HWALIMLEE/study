import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from keras.models import Sequential
from keras.layers import Dense, Flatten

iris = load_iris()
x = iris.data[:,(2,3)] #꽃잎의 길이와 너비
y = (iris.target==0).astype(np.int)

per_clf = Perceptron()
per_clf.fit(x,y)

y_pred = per_clf.predict([[2,0.5]])
