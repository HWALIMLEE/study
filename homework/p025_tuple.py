#튜플은 변경할 수 없는 리스트 /리스트는 변경 가능
my_list=[1,2]
my_tuple=(1,2)
other_tuple=3,4
my_list[1]=3

try:
    my_tuple[1]=3
except TypeError:
    print("cannot modify a tuple")

#함수에서 여러 값을 반환할 때 튜플 사용하면 편리
def sum_and_product(x,y):
    return (x+y),(x*y)

sp=sum_and_product(2,3) #(5,6)
s,p=sum_and_product(5,10) #s=15, p=50

#튜플과 리스트는 다중 할당
x,y=1,2
x,y=y,x #이제 x는 2, y는 1

