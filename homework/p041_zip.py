# list1=['a','b','c']
# list2=[1,2,3]

# pair for pair zip(list1,list2)

#주어진 리스트의 길이가 서로 다른 경우 zip은 첫 번째 리스트가 끝나면 멈춘다.
pairs=[('a',1),('b',2),('c',3)]
letters,numbers=zip(*pairs) #==letters,numbes=zip(('a',1),('b',2),('c',3))
# '*'는 인자 어패킹을 할 때 사용되는 문법으로, 이를 사용하면 pairs안의 항목들을 zip함수에 개별적인 인자로 전달

def add(a,b):
    return a+b

add(1,2)
try:
    add([1,2])
except TypeError:
    print("add expects two inputs")

print(add(*[1,2])) 

