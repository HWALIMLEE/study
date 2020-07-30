# 가운데 글자 가져오기
def solution(s):
    if len(s)%2==1:
        answer = s[len(s)//2]
    else:
        answer = s[len(s)//2-1]+s[len(s)//2]
    return answer

# 나누어 떨어지는 숫자 배열
def solution(arr, divisor): 
    return sorted([n for n in arr if n%divisor == 0]) or [-1]