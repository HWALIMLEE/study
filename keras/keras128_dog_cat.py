# 개랑 고양이 분류
# 남이 잘만든거 가져다 쓰기
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

img_dog = load_img('./data/dog_cat/dog.jpg',target_size=(224,224))
img_cat = load_img('./data/dog_cat/cat.jpg',target_size=(224,224))
img_yang = load_img('./data/dog_cat/yang.jpg',target_size=(224,224))
img_suit = load_img('./data/dog_cat/suit.jpg',target_size=(224,224))

plt.imshow(img_yang)
plt.imshow(img_cat)
# plt.show()

# 이미지를 데이터화
from keras.preprocessing.image import img_to_array, array_to_img #numpy로 변경
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_yang = img_to_array(img_yang)
arr_suit = img_to_array(img_suit)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

print("######################################")

# RGB ===> BGR (1번째와 3번째의 순서가 바뀐다, 전처리 하고 나서)
# vgg16 자체 전처리
# 이미지를 vgg16에 잘 넣기 위해서
# standardscaler 형식의 전처리
from keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_yang = preprocess_input(arr_yang)
arr_suit = preprocess_input(arr_suit)

print(arr_dog)
print(arr_dog.shape) #(224, 224, 3)

# 이미지 데이터를 하나로 합친다.
import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_yang,arr_suit])

print(arr_input.shape) #(4, 224, 224, 3)

# 모델 구성
model = VGG16() # 이미 가중치까지 정해져 있음
probs = model.predict(arr_input)

print(probs)
print('probs.shape:',probs.shape) # probs.shape: (4, 1000) // 총 4개(dog, cat, yang, suit)

# 이미지 결과
# 리스트 형태를 하나씩 출력하기 위해서
# 배경이 있음에도 불구하고 판단 가능
# 판단할 수 있는 사물은 1000개 정도
from keras.applications.vgg16 import decode_predictions
result = decode_predictions(probs)
print('============================================')
print(result[0])
print(result[1])
print(result[2])
print(result[3])
