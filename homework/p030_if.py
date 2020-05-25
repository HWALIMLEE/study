if 1>2:
    message="if only 1 were greater than two..."

elif 1>3:
    message="elif stands for 'else if'"

else:
    message="when all else fails use else(if you want to)"

#if-then-else문 한 줄로 표현
x=17
parity="even" if x %2==0 else "odd"
print(parity)

#while
x=0
while x<10:
    print(f"{x} is less than 10") #f-string 두 문자열 합치기
    x+=1
#더 복잡한 논리 체계(continue, break)
for x in range(10):
    if x==3:
        continue
    if x==5:
        break
    print(x)