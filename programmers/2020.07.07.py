# 문자열 다루기 기본
def solution(s):
    if len(s)==4 or len(s)==6:
        for i in range(len(s)):
            try:
                float(s[i])
            except:
                return False
            else:
                continue
        return True
    else:
        return False

# 서울에서 김서방 찾기
def solution(seoul):
    return "김서방은 " + str(seoul.index('Kim')) + "에 있다"
    