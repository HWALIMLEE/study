# 이상한 문자 만들기




# 자릿수 더하기
def solution(n):
    answer=[]
    n = str(n)
    for i in n:
        i = int(i)
        answer.append(i)
    return sum(answer)

