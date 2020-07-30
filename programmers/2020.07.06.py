#문자열 내 p와 y의 개수
def solution(s):
    p_count=0
    y_count=0
    for i in s:
        if i.upper()=='P':
            p_count+=1
        if i.upper()=='Y':
            y_count+=1
    if p_count==y_count:
        return True
    elif p_count==y_count==0:
        return False
    else:
        return False

# 문자열 내림차순으로 배치
def solution(s):
    return ''.join(sorted(s,reverse=True))
# join은 리스트에서 문자열로
# 원래 대문자가 먼저 오기 때문에 reverse=True써주면 같은 문자 안에서는 소문자 먼저 오게 되고 문자도 큰것부터 작은 순으로 정렬시켜줌