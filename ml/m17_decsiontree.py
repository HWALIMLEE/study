from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(
    cancer.data, cancer.target, train_size=0.9, random_state=42)

model = DecisionTreeClassifier(max_depth=4) #max_depth=3이나 4정도가 적정하다고 알고 있다

model.fit(x_train,y_train)

acc = model.score(x_test,y_test)

print("acc:",acc)

print(model.feature_importances_) #feature_importances_ //컬럼 30개 중에 8번째 컬럼이 accuracy에 영향을 많이 끼친다
# 가중치 줄 때 유리하겠군
# 혹은 PCA쓸때

print(x_train.shape)