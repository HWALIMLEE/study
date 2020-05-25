def generate_range(n):
    i=0
    while i<n:
        yield i
        i+=1

for i in generate_range(10):
    print(f"i:{i}") #i:0/i:1/i:2/.....

#무한한 수열도 메모리의 제약을 받지 않고 구현할 수 있다.
def natural_numbers():
    """1,2,3,...을 반환"""
    n=1
    while True:
        yield n
        n+=1
#물론 break없이 무한 수열 생성하는 것은 추천하지 않는 방법

print(natural_numbers())

#괄호 안에 for문을 추가하는 방법으로 제너레이터 만들기
evens_below_20=(i for i in generate_range(20) if i%2==0)

print(evens_below_20)

#enumerate함수(순서, 항목)형태로 값을 반환
names=["Alice","Bob","Charlie","Debbie"]

#파이썬스럽지 않다.
for i in range(len(names)):
    print(f"name {i} is {names[i]}")
print("===========")
#파이썬스럽지 않다.
i=0
for name in names:
    print(f"name {i} is {names[i]}")
    i+=1
print("====================")
#파이썬스럽다
for i,name in enumerate(names):
    print(f"name {i} is {name}")