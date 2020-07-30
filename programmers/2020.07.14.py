# 콜라츠 추측
def solution(num):
    count = 0
    if num==1:
        return 0
    else:
        while count < 501:
            if num%2==0:
                a = num//2
                if a == 1:
                    count += 1
                    return count
                else:
                    num = a
                    count += 1
                    pass
            else:
                b = num*3 + 1
                if b == 1:
                    count += 1
                    return b
                else:
                    num = b
                    count += 1
                    pass
            if count > 500:
                return -1


# 명석쓰 코드
def solution(num):
    answer = 0
    while (num != 1):
        if(num%2 == 0):
            num //= 2
            answer +=1
        else :
            num = num*3 + 1
            answer +=1
        if(answer>=500):
            answer = -1
            break
    return answer


# 평균 구하기
def solution(arr):
    return sum(arr)/len(arr)
