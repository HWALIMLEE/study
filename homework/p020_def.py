def double(x):
    return x*2


def apply_to_one(f):
    """
    인자가 1인 함수를 호출
    """
    return f(1)

my_double=double
x=apply_to_one(my_double)
print(x)

#람다함수
y=apply_to_one(lambda x:x+4)
print(y)

another_double=lambda x:2*x #이 방법은 최대한 피할 것
print(x)

"""
def another_double(x):
    return 2*x
print(another_double(apply_to_one(my_double)))
"""

def my_print(message="my default message"):
    print(message)

my_print("hello") #hello
my_print() #default메시지 출력(my default message)


def full_name(first="What's-his-name",last="Something"):
    return first+" "+last
print(full_name("Joel","Grus")) #Joel Grus
print(full_name("Joel")) # Joel Something
print(full_name(last="Grus")) #What's-his-name Grus