def double(x):
    return x*2
print(double(5))

def apply_to_one(f):
    return f(1)

x=apply_to_one(double)
print(x)

y=apply_to_one(lambda x:x+4) #변수에 람다 함수 할당

another_double=lambda x:2*x

def another_double(x):
    return 2*x

def my_print(message="my default message"):
    print(message) 
my_print("hello")
my_print() #my default message

def full_name(first="Wath's-his-name",last="Something"):#인자 이름 명시
    return first+" "+last
print(full_name("Joel","Grus"))
print(full_name("Joel"))
print(full_name(last="Grus"))

