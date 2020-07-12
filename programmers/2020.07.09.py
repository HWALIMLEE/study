# 시저 암호
def solution(s, n):
    answer=[]
    for i in range(len(s)):
        if s[i]==' ':
            answer.append(' ')
        elif ord(s[i])>=65 and ord(s[i])<=90: #대문자
            x = ord(s[i])+n
            if x > 90 and x < 116:
                a = chr(x-26)
                answer.append(a)
            else:
                answer.append(chr(x))
        elif ord(s[i])>=97 and ord(s[i])<=122: #소문자
            x = ord(s[i])+n
            if x > 122:
                x = chr(x-26)
                answer.append(x)
            else:
                x = chr(x)
                answer.append(x)
    return "".join(answer)


# 명석쓰 코드
def solution(s, n):
    answer = ''
    s_list = list(s) # 리스트 변환
    for i in range(len(s_list)):
        if s_list[i] == ' ':
            answer += ' '
        if s_list[i].isupper(): # 대문자일 때 true return
            answer += chr((ord(s_list[i])- ord('A') + n)%26 + ord('A') )
        
        elif s[i].islower(): # 소문자일 때 true return
            answer += chr((ord(s_list[i])- ord('a') +n)%26 + ord('a') )
    return answer



# 약수의 합 구하기
def solution(n):
    answer=[]
    for i in range(1, n+1):
        if n%i==0:
            answer.append(i)
    return sum(answer)

# 민지쓰 코드
def solution(n):
    answer = [i for i in range(1,n+1) if n%i==0]
    return sum(answer)