# 직사각형 별찍기
a, b = map(int, input().strip().split(' '))
i=0
while i < b:
    print("*"*a , sep='\n') 
    i+=1

# 예산
def solution(d, budget):
    d = sorted(d)
    a = d[0]
    i = 1
    if a > budget:        # 아무도 지원 받지 못할 때
        return 0 
    if sum(d) <= budget:  # 모두 지원 받을 수 있을 때
        return len(d)
    else:                 # 일부만 지원 받을 수 있을 때
        while i < len(d):
            a = a + d[i]
            if a <= budget:
                i+=1
                continue
            else:
                result=i
                break
    return result

# 다른 풀이 - 너무 신기했음, pop()뒤에꺼부터 하나씩 제거
def solution(d, budget):
    d.sort()
    while budget < sum(d):
        d.pop()
    return len(d)
