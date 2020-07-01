# 완주하지 못한 선수
def solution(participant,completion):
    participant.sort()
    completion.sort()
    for p, c in zip(participant, completion): # zip문을 써서 하나씩 받아옴 
        if p!=c:
            return p                          # 순서대로 정렬한 거에서 서로 같지 않으면 p값 반환
    return participant[-1]                    # 모두 같고 끝에 사람이 다를 경우 감안

# 크레인 인형뽑기
def solution(board, moves):
    answer = 0
    b = []
    c = []
    a = [list(x) for x in zip(*board)] # 리스트 전치 [[0,0,0,4,3],[0,0,2,2,5],[0,1,5,4,1],[0,0,0,4,3],[0,3,1,2,1]]
    for i in moves:
        for j in a[i-1]:
            if j==0:                   # 0이면 패스
                pass 
            else:                      # 한 번 뽑힌 숫자 삭제
                b.append(j)
                a[i-1].remove(j)   
                break                  #여기까지 b=[4,3,1,1,3,2,4] 생성
    while True:
        count =0
        L = len(b)                      # 개수 갱신시켜주기 위해서(밑에서 삭제한 후 다시 갱신)
        for n in range(L-1):
            if b[n]==b[n+1]:           
                c.append(b[n])          # 연속된 숫자 있으면 새로운 빈 리스트 c 에 추가
                del b[n:n+2]            # 연속된 숫자 추가한 거 b리스트에서 지우기
                break                   # break문 써줘서 다시 while True:로 돌아감 
            if n == L-2:                # 끝까지 갔을 때에 연속된 숫자 없으면 count=1로 지정해 준 후
                count =1 
        if count == 1:                  # 연속된 숫자 없기 때문에 while문 종료
            break
    return len(c)*2