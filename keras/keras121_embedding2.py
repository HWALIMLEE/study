from keras.preprocessing.text import Tokenizer
import numpy as np

# 가장 긴 길이는 5개
docs = ["너무 재밌어요","참 최고에요","참 잘 만든 영화에요",
        "추천하고 싶은 영화입니다.","한 번 더 보고 싶네요","글쎄요",
        "별로에요","생각보다 지루해요","연기가 어색해요",
        "재미없어요","너무 재미없다","참 재밌네요"]

# 긍정1, 부정0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs) # 인덱싱 형태로 제공
print(token.word_index)  # 단언별로 인덱스 줌
# {'너무': 1, '참': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고': 13, '싶네요': 14, 
# '글쎄요': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21, '재미없다': 22, '재밌네요': 23}
#  동일한 단어 반복되서 안나옴
#  두번이상 반복되는 단어가 앞에 나옴
#  많이 사용하는 단어부터 앞에 나옴
#  참 3번, 너무 2번나오면 참이 맨 앞에 나옴

# 전체 문장에 대한 인덱스
x = token.texts_to_sequences(docs) 
print(x)   # [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23]]
# 문제점
# >>> shape가 모두 틀림
# shape 맞춰주어야함// padding-빈자리에 0을 넣어서 shape 맞춰줬었음
# 최대 큰 거에 맞춰서 나머지 다 0채워줌


from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding="pre") 
# pad_x = pad_sequences(x, padding="post",value=1)
print(pad_x)   # (12, 5)
### padding='pre' 0이 앞에서부터 들어간다// padding='post' 0이 뒤에서부터 들어간다// value 넣을 수도 있다(0대신 value값 들어감)
### word.index ===> texts_to_sequence ===> pad_sequences
### 데이터 준비 되었으니 모델 짜기(embedding으로)

### <embedding>
# >> 단어별로 공통된 부분 벡터화한다(원핫인코딩의 압축형)


word_size = len(token.word_index) + 1
print("전체 토큰 사이즈:", word_size)  #25

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

model = Sequential()
# model.add(Embedding(word_size, 10, input_length=5)) # 이것만 기억!/ 와꾸 맞춰주는 부분 (25, 임의의 수(히든), 5)=(전체 단어의 개수, 아웃풋, 5) <===  (12,5)(행 무시) #(None, 5, 10)=(None, input_length, output)
# word_size 부분을 임의의 숫자로 바꾸면
# >>> 잘 돌아간다
# >>>임의의 숫자로 바꿀 수 있다. 전체 단어 크기 제대로 주는 게 제일 좋다. 그러나 임의의 숫자 주었다고 해서 틀린 건 아니다. (통상적으로는 단어 개수만큼)
# >>>단지 틀려지는 부분은 parameter 개수
# 그 다음에 전달하는 출력노드의 개수 = 10

# embedding 은 2차원으로 바꾸는 것
model.add(Embedding(word_size, 10)) # input_length(열과 동일) 제거. 
# The shape of the input to "Flatten" is not fully defined (got (None, 10)). Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.

# model.add(Flatten())  #Flatten 하면 Dense 바로 붙일 수 있음
model.add(Dense(1,activation='sigmoid')) # 최종 출력은 '긍정, 부정' 

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model.fit(pad_x,labels,epochs=30)

acc = model.evaluate(pad_x, labels)[1] 
# [1] 의 의미는?
# >>> metrics값을 빼겠다 (evaluate 는 loss값, metrics값 반환)
print("acc:",acc)

"""
순환 신경망 모델
문장을 단어들의 시퀀스로 간주하고 순환(LSTM) 레이어의 입력으로 구성한 모델입니다.
임베딩 레이어 다음에 LSTM 레이어가 오는 경우에는 임베딩 레이어에 input_length 인자를 따로 설정할 필요는 없습니다. 
입력 문장의 길이에 따라 input_length가 자동으로 정해지고, 이것이 LSTM 레이어에는 timesteps으로 입력되기 때문입니다. 
블록으로 표현한다면 예제에서는 문장의 길이가 200 단어이므로, LSTM 블록 200개가 이어져있다고 생각하면 됩니다.
"""

