import heapq
def solution(scoville, K):
    heapq.heapify(scoville)
    result=0
    while len(scoville) >=2: #IndexError방지
        min1 = heapq.heappop(scoville)
        if min1>=K:
            return result
        else:
            min2 = heapq.heappop(scoville)
            heapq.heappush(scoville, min1+2*min2)
            result+=1
    if scoville[0] > K:
        return result
    else:
        return -1
