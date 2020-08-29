# 소수찾기
# 시간초과, 효율성 테스트 시간초과
def solution(n):
    answer=[]
    aaa = [a for a in list(range(2,n+1)) if a%2!=0]
    i=1
    while i < len(aaa):
        k=i-1
        while 0 <= k < len(aaa)-1:
            if aaa[i]%aaa[k]==0:
                answer.append(aaa[i])
                break
            else:
                k-=1
        i+=1
    return len(set(aaa)-set(answer))+1

# 에라토스테네스의 체 이용
def solution(n):
    sieve = [True] * (n+1)

    # n의 최대 약수가 sqrt(n) 이하이므로 i=sqrt(n)까지 검사
    m = int(n ** 0.5)
    for i in range(2, m + 1):
        if sieve[i] == True:           # i가 소수인 경우
            for j in range(i+i, n+1, i): # i이후 i의 배수들을 False 판정
                sieve[j] = False

    # 소수 목록 산출
    return len([i for i in range(2, n+1) if sieve[i] == True])
