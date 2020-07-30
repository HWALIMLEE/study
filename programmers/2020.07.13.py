# 짝수와 홀수
def solution(num):
    if num%2:
        return "Odd" # 1이면 True이고 if문 통과함
    else:
        return "Even" #False이면 0

# 최대공약수와 최소공배수
def solution(n, m):
    answer=[]
    if m % n==0:
        return [n,m] # 서로가 나누어 떨어지는 수 일때 
    for i in range(n,0,-1): # 거꾸로 배열시킨 다음(최대 공약수 찾기 위해)
        if n % i==0 and m % i==0: # 둘 다 나누어 떨어지는 수 찾기
            answer.append(i)      # 최대 공약수 추가
            answer.append(i*(n//i)*(m//i))  # 최소 공배수 추가
            break # 한 번 하고 빠져 나오기
    return answer
   
