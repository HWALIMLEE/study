# K번째 수
def solution(array, commands):
    answer=[]
    for a in commands:
        array2 = array[a[0]-1:a[1]]
        array2.sort()
        answer.append(array2[a[2]-1])
    return answer

# 모의고사 - 예제는 맞았지만 다른 것이 틀렸음
def solution(answers):
    import numpy as np
    import operator
    a_score = 0
    b_score = 0
    c_score = 0
    box=[]
    a = [1,2,3,4,5]*2000
    b = [2,1,2,3,2,4,2,5]*2000
    c = [3,3,1,1,2,2,4,4,5,5]*1000
    answers = answers*2000
    b = b[:10000]
    # a 와 answer비교해서 같으면 True, 다르면 False반환
    a_answers = np.equal(a,answers)  
    b_answers = np.equal(b,answers)
    c_answers = np.equal(c,answers)
    # 각각의 score반환
    for a in a_answers:
        if a==True:
            a_score+=1              
    box.append(a_score)
    for b in b_answers:
        if b==True:
            b_score+=1
    box.append(b_score)
    for c in c_answers:
        if c==True:
            c_score+=1
    box.append(c_score)
    # 중복되는 수 제거하고 반환
    box_set = list(set(box))
    if len(box_set)>1:
        return [box_set.index(max(box_set))+1]
    else:
        return [1,2,3]
    # 문제점은 마지막에 두명의 점수가 같고 한명이 다를 때를 고려하지 못했음
    # 최고점이 한명 있을 때와 모두 점수가 같을 때 밖에 식을 세우지 못했음


# 민지님 코드로 다시 공부
def solution(answers):
    a_score = 0
    b_score = 0
    c_score = 0
    box=[]
    answer = []
    a = [1,2,3,4,5]*2000
    b = [2,1,2,3,2,4,2,5]*2000
    c = [3,3,1,1,2,2,4,4,5,5]*1000
    length = len(answers)
    a = a[:length]
    b = b[:length]
    c = c[:length]
    # num, count로 반환하니 인덱스와 점수를 동시에 불러올 수 있었음
    num = 0
    for person in [a,b,c]:
        count=0
        num+=1
        for i, k in zip(person,answers):
            if i==k:
                count+=1
        box.append([num,count])
    # lambda의 key값으로 정렬
    box.sort(key=lambda x : x[1], reverse=True)
    for i in box:
        if box[0][1]==i[1]:
            answer.append(i[0])
    return answer   


