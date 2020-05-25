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


