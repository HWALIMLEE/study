# 자연수 뒤집어 배열로 만들기





# 정수 내림차순으로 배치하기
def solution(n):
    answer = []
    n = str(n)
    for i in n:
        answer.append(i)
    return int("".join(sorted(answer,reverse=True)))