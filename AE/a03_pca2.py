import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print(X.shape) # (442,10)
print(Y.shape) # (442,)

pca = PCA(n_components=5)
x2 = pca.fit_transform((X))
pca_evr = pca.explained_variance_ratio_
print(pca_evr) # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856] # 5개의 컬럼으로 압축(압축한 컬럼들의 중요도)
print(sum(pca_evr)) # 0.83 
# 1이 안나오는 이유는?
# >> n_components가 원래 10개니까 10으로 줬을 때 1
# 다시 원복해도 99.9%나올 수 있음
