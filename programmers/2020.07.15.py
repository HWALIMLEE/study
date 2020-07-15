# 하샤드 수
def solution(x):
    p = 0
    xs = str(x)
    for i in xs:
        i = int(i)
        p+=i
    if x % p==0:
        return True
    else:
        return False

# 다른 풀이
def solution(x):
    return x % sum([int(c) for c in str(x)])==0

# 전화번호 가리기
def solution(phone_number):
    phone_number_reverse = list(reversed(phone_number)) # sorted쓸때는 오류 났는데 list(reversed())쓰니까 오류 나지 않는다 
    for i in range(len(phone_number_reverse)):
        if i > 3:
            phone_number_reverse[i]="*"
    return "".join(list(reversed(phone_number_reverse)))
