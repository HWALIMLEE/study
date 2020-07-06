from keras.preprocessing.text import Tokenizer

text = "나는 맛있는 밥을 먹었다"

####단어별###
token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)  # {'나는': 1, '맛있는': 2, '밥을': 3, '먹었다': 4} ===> 단지 인덱싱
# 수치화 ===> 자연어 처리의 기반
# Token화 ===> 한 개 문장을 단어단어로 자르고 인덱싱 걸어줌

###문자를 순서로 정한다.###
x = token.texts_to_sequences([text])
print(x)    # [[1, 2, 3, 4]]

## 모든 단어 수치화 시키고 모델 돌리기
## 인덱싱 ===> 범주화 (밥을:3이 나는:1에 대해 3배가 아니기 때문)
## 원핫 인코딩
from keras.utils import to_categorical

word_size = len(token.word_index) +1 ## to_categorical은 0부터 시작하기 때문에 1더해줌
x = to_categorical(x,num_classes=word_size)
print(x)  ## 4*5 행렬

## OneHotEncoder써도 된다

### 범주화의 문제점은 사이즈가 너무 커진다는 것이다. 모든 단어 범주화 시키기 때문에
### 그래서 압축이 필요함(embedding)
### embedding은 시계열에서도 많이 쓴다