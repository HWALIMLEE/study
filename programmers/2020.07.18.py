# 비밀지도
def solution(n, arr1, arr2):
    answer = []
    answer2 = []
    answer3 = []
    for i, j in zip(arr1, arr2):
        answer.append((i|j))
    for a in answer:
        answer2.append(bin(a)[2:].zfill(n))
    for b in answer2:
        for k in b:
            if k=='1':
                b = b.replace(k,"#")
            else:
                b = b.replace(k," ")
        answer3.append(b)
    return answer3

# 실패율====>실패했음(시간초과)
def solution(N, stages):
    answer=[]
    stages = sorted(stages)
    for i in range(1,N+1):
        for j in stages:
            if i>=j:
                answer.append(stages.count(j)/len(stages))
                while stages.count(j) > 0:
                    stages.remove(j)
            else:
                answer.append(0)
            break   
    result = [i[0]+1 for i in sorted(enumerate(answer),key=lambda x:x[1],reverse=True)]
    if max(result) < N:
        s = list(range(max(result)+1,N+1))
        result = result + s
    else:
        pass
    return result

# 답지 참고...
def solution(N, stages):
    result = {}
    denominator = len(stages)
    for stage in range(1, N+1):
        if denominator != 0:
            count = stages.count(stage)
            result[stage] = count / denominator
            denominator -= count
        else:
            result[stage] = 0
    return sorted(result, key=lambda x : result[x], reverse=True)