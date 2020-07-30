# 두 정수 사이의 합
def solution(a, b):
    answer = 0
    if a < b:
        answer = answer + a*(b-a+1) + (b-a)*(b-a+1)/2  # ex) (3+0)+(3+1)+(3+2)===> 3이 총 3개 + 0부터 2까지 합은 가우스 정리 이용
    elif a > b:
        answer = answer + b*(a-b+1) + (a-b)*(a-b+1)/2
    else:
        answer = a
    return answer

# 문자열 내 마음대로 정렬
def solution(strings, n):
    strings.sort()
    answer = sorted(strings, key=lambda x:x[n])
    return answer

# 먼저 정렬한 후 lambda이용
