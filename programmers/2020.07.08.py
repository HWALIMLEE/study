# 소수찾기



#문자열을 정수로 바꾸기
def solution(s):
    answer = 0
    if s.startswith(""):
        answer = int(s)
    elif s.startswith("-"):
        s[0].replace("-","")
        answer = - + int(s)
    return answer