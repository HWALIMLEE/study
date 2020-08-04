import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print(X.shape) # (442,10)
print(Y.shape) #(442,)

# pca = PCA(n_components=5)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_
# print(pca_evr) # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856] # 5개의 컬럼으로 압축(압축한 컬럼들의 중요도)
# print(sum(pca_evr)) 

# 0.83 
# 0.17의 손실을 보았다 그렇지만, mnist에서 0이 차지하는 부분이 80% 라면, 그것을 빼고 압축이 되어도 상관이 없다

# 1이 안나오는 이유는?
# >> n_components가 원래 10개니까 10으로 줬을 때 1
# 다시 원복해도 99.9%나올 수 있음


# n_componenets 명시하지 않음
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_) # cumsum은 누적합 구하는 함수 [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.]
print(cumsum)

print(np.argmax(cumsum >= 0.94)+1) # 0.94이상 나오게 하려면 7개 이상 써야 한다. 
print(cumsum>=0.94) # [False False False False False False  True  True  True  True]

