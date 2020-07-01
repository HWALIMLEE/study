# 1. 같은 숫자는 싫어
def solution(arr):
    answer=[]
    if arr[0]==arr[1]:
        answer.append(arr[0])
        i=2
        while i < len(arr):
            if arr[i]!=arr[i-1]:
                answer.append(arr[i])
            elif arr[i]==arr[i-1]:
                pass
            i+=1
    elif arr[0]!=arr[1]:
        answer.append(arr[0])
        i=1
        while i<len(arr):
            if arr[i]!=arr[i-1]:
                answer.append(arr[i])
            elif arr[i]==arr[i-1]:
                pass
            i+=1
    return answer

# 2. 수박수박수박수
def solution(n):
        if n%2==0:
            answer="수박"*(n//2)
        else:
            answer="수박"*(n//2) +"수"
        return answer 