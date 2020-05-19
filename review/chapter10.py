import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# 퍼셉트론이 동작하는 방식은 다음과 같다.
# 각 노드의 가중치와 입력치를 곱한 것을 모두 합한 값이 활성함수에 의해 판단되는데, 
# 그 값이 임계치(보통 0)보다 크면 뉴런이 활성화되고 결과값으로 1을 출력한다. 뉴런이 활성화되지 않으면 결과값으로 -1을 출력한다.

