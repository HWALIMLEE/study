# 2. 튜플
# 리스트와 거의 같으나, 삭제, 수정이 안된다. (변경이 안되는 값)
a=(1,2,3) #--->고정값에만 씀 #튜플
b=1,2,3 #튜플
print(type(a))
print(type(b))

# a.remove(2)
# print(a)  --->AttributeError(속성에러)

#개체 안이 삽입 삭제 수정이 안되는 것, 이 연산은 가능하다. 
print(a+b)
print(a*3)

# print(a-3)---->수정 안된다.(typeError)

#가장 많이 쓰는 건 리스트

