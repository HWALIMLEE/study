# 정수형
a=1
b=2
c=a+b
print("c:",c)
d=a*b
print("d:",d)
e=a/b #정수와 정수를 나눴을 때 실수 출력
print("e:",e)

#실수형
a=1.1
b=2.2
c=a+b
print("c: {:.2f}".format(c))

d=a*b
print("d: {:.2f}".format(d))

e=a/b
print("e:",e)

#문자형
a='hel'
b='lo'
c=a+b #연결이 된다. 
print("c:",c) #hello

#문자+숫자
# a=123
# b='45'
# c=a+b 
# print("c:",c) #TypeError가 난다. 

#숫자를 문자로 변환+문자  (형변환)
a=123
a=str(a)
print("a:",a)
b='45'
c=a+b
print(c)

#문자를 숫자로 변환+숫자  (형변환)

a=123
b='45'
b=int(b)
print("b:",b)
c=a+b
print("c:",c)

#문자열 연산하기
a='abcdefgh'
print(a[0])
print(a[5])
print(a[-1])
print(a[-2])
print(type(a))

b='xyz'
print(a+b)

#문자열 인덱싱
a="Hello, Deep learning" #, ' ' 문자
print(a[7])
print(a[3:9])
print(a[3:-5])
print(a[5:-4])
print(a[:-1])