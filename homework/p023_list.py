integer_list=[1,2,3]
heterogeneous_list=["string",0.1,True]
list_of_lists=[integer_list,heterogeneous_list,[]]

list_length=len(integer_list)
list_sum=sum(integer_list)
print(list_length)
print(list_sum)

#리스트의 n번째 값 불러오기
x=[0,1,2,3,4,5,6,7,8,9]
zero=x[0]
one=x[1]
nine=x[-1]
eight=x[-2]
x[0]=-1 #--->대체

print(x) #[-1,1,2,3,4,6,7,8,9]

#슬라이싱
first_three=x[:3] #[-1,1,2]
three_to_end=x[3:] #[3,4,5,6,7,8,9]
one_to_four=x[1:5] #[1,2,3,4]
last_three=x[-3:] #[7,8,9]
without_first_and_last=x[1:-1] #[1,2,3,4,5,6,7,8]
copy_of_x=x[:] #[-1,1,2,3,4,5,6,7,8,9]

#간격 설정하여 분리
every_third=x[::3]
five_to_three=x[5:2:-1] #[a:b:c]--->range(a,b) 간격 c
print(five_to_three)

#in연산자
a=1 in [1,2,3] #True
b=0 in [1,2,3] #False

print(a)
print(b)

#리스트 추가
x=[1,2,3]
x.extend([4,5,6])
print(x)  #[1,2,3,4,5,6]

#리스트에 항목 추가
x=[1,2,3]
x.append(0) #[1,2,3,0]
y=x[-1] #0
z=len(x) #4

#리스트 풀기
x,y=[1,2]
print(x)
print(y)

#버릴 항목은 밑줄(under_bar)로 표시
_,y=[1,2]
print(y)


