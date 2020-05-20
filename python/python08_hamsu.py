##함수
def sum(a,b):
    return a+b

print(sum(3,4))


###곱셈, 나눗셈, 뺄셈 함수를 만드시오
#mull, div1, sub1
#매개변수(parameter)=(a,b)
def mull(a,b):
    return a*b

def div1(a,b):
    return a/b

def sub1(a,b):
    return a-b


print("div1:",div1(5,2))
print("mull:",mull(5,2))
print("sub1:",sub1(5,2))

#매개변수 없는 함수 
def sayYeh():
    return "Hi"
aaa=sayYeh()
print(aaa)

#매개변수 3개

def sum1(a,b,c):
    return a+b+c

a=1
b=2
c=34
d=sum1(a,b,c)
print("d:",d)