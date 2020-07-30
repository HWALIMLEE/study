#lambda는 쓰고 버리는 일시적인 함수
#함수가 생성된 곳에서만 필요
#즉, 간단한 기능을 일반적인 함수와 같이 정의해두고 쓰는 것이 아니고, 필요한 곳에서 즉시 사용하고 버림

a=[1,2,3,4]
b=[17,12,11,10]
print(list(map(lambda x,y:x+y,a,b)))

#람다 정의에는 "return"문이 포함되어 있지 않습니다.
#함수가 사용될 수 있는 곳에는 어디라도 람다 정의를 넣을 수 있습니다.

g=lambda x:x**2
print(g(8))
f=lambda x,y:x+y
print(f(4,4))

def inc(n):
    return lambda x:x+n

f=inc(2)
print(f(12)) #14

g=inc(4)
print(g(12)) #16

print(inc(2)(12)) #14


#람다함수의 장점은 map()함수와 함께 사용될 때 볼 수 있다.
#1. map()은 두 개의 인수를 가지는 함수
#r=map(function,iterable...)
#function은 함수의 이름, iterable은 한 번에 하난의 멤버를 반환할 수 있는 객체

a=[1,2,3,4]
b=[17,12,11,10]
map_answer=list(map(lambda x,y:x+y,a,b))
print("answer:",map_answer)

#2. filter()함수도 두개의 인자를 가진다.
#r=filter(function,iterable)
#filter에 인자로 사용되는 function은 처리되는 각각의 요소에 대해 boolean값을 반환합니다. 
#True로 반환하면 그 요소는 남게 되고, False를 반환하면 그 요소는 제거 됩니다.

#3의 배수인 것만 남기기
foo=[2,18,9,22,17,24,8,12,27]
foo_answer=list(filter(lambda x:x%3==0,foo)) #[18,14,14,14]
print("answer:",foo_answer) #[18,9,24,12,27]

#3. reduce()함수-내장함수 아님
#reduce()함수를 두 개의 필수 인자와 하나의 옵션 인자를 가진다.
#function을 사용해서 iterable을 하나의 값으로 줄인다.
#initializer는 주어지면 첫 번째 인자로 추가 된다고 생각
#functools.reduce(function,iteralbe[,initializer])

from functools import reduce
reduce_answer=reduce(lambda x,y:x+y,[1,2,3,4,5])
print("answer:",reduce_answer) #15

# https://offbyone.tistory.com/73--->참조


#if를 이용한 람다
def lower_three1(x):
    if x<3:
        return x*2
    else:
        return x/3+5
    
lower_three2=lambda x:x*2 if x<3 else x/3+5
#이처럼 람다식을 이용하면 코드 절약할 수 있다.

#람다를 이용하여 a가 10이상 30미만이면 a**2-40a+350의 계산값을, 그 이외의 경우에는 50을 반환
a1=13
a2=32
func5=lambda a:a**2-40*a+350 if a>=10 and a<30 else 50
print(func5(15))

