#자료형
#1. 리스트--->여러가지 자료형을 쓸 수 있다. 
a=[1,2,3,4,5]
b=[1,2,3,'a','b']
print(b)

# numpy는 한가지 자료형만
print(a[0]+a[3])
# print(b[0]+b[3]) #--->TypeError
print(type(a))
print(type(b[3]))

#1-2.리스트 슬라이싱
a=[1,2,3,['a','b','c']]
print(a[1])#2
print(a[-1])#[a,b,c]
print(a[-1][1]) #b

a=[1,2,3,4,5]
print(a[:2]) #[1,2]


#1-3. 리스트 더하기
a=[1,2,3]
b=[4,5,6]
print(a+b) #[1,2,3,4,5,6] 가중치 연산(행렬 연산)

#numpy.array는 a+b=[5,7,9] 사람이 하는 연산처럼 해줌/ numpy 속도 매우 빠름/ 같은 타입만 쓸 수 있다. 
#numpy쓸 때 list 제일 중요
# import numpy as np
# a=np.array([1,2,3])
# b=np.array([4,5,6])
# print(a+b) #[5 7 9]

c=[7,8,9,10]
print(a+c)
print(a*3) #[1,2,3,1,2,3,1,2,3]

# a=np.array([1,2,3])
# print(a*3)--->[3 6 9]

print(str(a[2])+'hi')

f='5'
# print(a[2]+f) #--->오류
print(str(a[2])+f)
print(a[2]+int(f))

# append가 제일 중요
# 리스트 관련 함수
a=[1,2,3]
a.append(4)
print(a)

# a=a.append(5)  오류: 다시 자기거에 집어넣을 수 없다. 그냥 a.append()라고 해야함

# sort
a=[1,3,4,2]
a.sort()
print(a)

# reverse
a.reverse()
print(a)

print(a.index(3))  # ==a[3]
print(a.index(1))  # ==a[1]

a.insert(0,7) #[7, 4, 3, 2, 1] #0번째 인덱스에 7추가
print(a)

a.insert(3,3) #[7,4,3,3,2,1] #3번째 인덱스에 3추가
print(a)

a.remove(7) #[4,3,3,2,1] remove는 인자값 지우는 것
print(a)

a.remove(3) #[4,3,2,1] 먼저 걸리는 원소 하나만 지워진다. 
print(a)

