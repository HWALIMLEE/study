def doubler(f):
    def g(x):
        return 2*f(x)
    return g

def f1(x):
    return x+1

g=doubler(f1)
assert g(3)==8, "(3+1)*2 should equal 8"
assert g(-1)==0, "(-1+1)*2 should equal 0"

#두 개 이상의 인자를 받는 함수의 경우 문제 발생
def f2(x,y):
    return x+y
g=doubler(f2)

try:
    g(1,2)
except TypeError:
    print("as defined, g only takes one argument")

def magic(*args, **kwargs):
    print("unnamed args:",args) #unnamed args:(1,2)
    print("keyword args:",kwargs) #keyword args:{'key':'word','key2':'word2'}

magic(1,2,key="word",key2="word2")

def other_way_magic(x,y,z):
    return x+y+z
x_y_list=[1,2]
z_dict={"z":3}
assert other_way_magic(*x_y_list,**z_dict)==6,"1+2+3 should be 6"

def doulber_correct(f):
    def g(*args,**kwargs):
        return 2*f(*args,**kwargs)
    return g

g=doubler_correct(f2)
assert g(1,2)==6, "doubler should work now"

