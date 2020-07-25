# 다트게임
def solution(dartResult):
    box=[]
    answer=0
    for i in dartResult:
        box.append(i)
        for i in range(len(box)-1):
            # 10일때 고려
            if box[i]=='1':
                if box[i+1]=='0':
                    box[i]=box[i]+box[i+1]
                    box.remove(box[i+1])
    for i in range(len(box)):
        if box[i]=='S':
            answer+=int(box[i-1])
        elif box[i]=='D':
            answer+=int(box[i-1])**2
        elif box[i]=='T':
            answer+=int(box[i-1])**3
        elif box[i]=='*':
            answer = answer*2
        elif box[i] == '#':
            if box[i-1]=='S':
                answer-=2*int(box[i-2])
            elif box[i-1]=='D':
                answer-=2*(int(box[i-2])**2)
            elif box[i-1]=='T':
                answer-=2*(int(box[i-2])**3)
    if box[-1]=='*':
        if box[1]=='S':
            answer -= int(box[0])
        elif box[1]=='D':
            answer -= int(box[0])**2
        elif box[1]=='T':
            answer -= int(box[0])**3
    return answer