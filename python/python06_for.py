a={'name':'yun','phone':'010','birth':'0511'}
for i in a.keys():
    print(i) #name \n phone \n birth

a = [1,2,3,4,5,6,7,8,9,10]
for i in a:
    i=i*i
    print(i)

for i in a:
    print(i)

##while문
"""
while조건문:        #조건이 참일 동안 계속 돈다.
    수행할 문장
"""

###if문
if 1:
    print('True')
else:
    print('False')

if 3:
    print('True')
else:
    print('False')

if 0:
    print('True')
else:
    print('False')

if -1:
    print('True')
else:
    print('False')

"""
비교연산자
<,>,==,!=,>=,<=
"""
# if a=1:
#     print("출력잘돼") #sytanxError
a=1
if a==1:
    print("출력잘돼")


money=10000
if money>=30000:
    print("한우먹자")
else:
    print("라면먹자")


### 조건연산자
#and, or, not
money=20000
card=1
if money>=30000 or card==1:
    print("한우먹자")
else:
    print("라면먹자")


#break,continue
print("==========break==========")
score=[90,25,67,45,80]
number=0
for i in score:
    if i<30:
        break #break당하면 제일 가까운 for문 중단시킴 #90만 나오게 된다.
    if i>=60:
        print("경축")
        number=number+1
print("합격인원:",number,"명")

print("==========continue=========")
score=[90,25,67,45,80]
number=0
for i in score:
    if i<60:
       continue #continue 문 걸리면 for문 다시 위로 올라감 밑에꺼 실행 안됨
    if i>=60:
        print("경축")
        number=number+1
print("합격인원:",number,"명")

