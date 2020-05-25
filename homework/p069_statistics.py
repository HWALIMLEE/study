#median
#홀수면 중앙값을 반환
def _median_odd(xs:List[float])->float:
    """len(xs)가 홀수면 중앙값을 반환"""
    return sorted(xs)[len(xs)//2] #'//'은 소수점 이하를 버리는 연산자

#짝수면 두 중앙값의 평균을 반환
def _median_even(xs:List[float])->float:
    sorted_xs=sorted(xs)
    hi_midpoint=len(xs)//2
    return (sorted_xs[hi_midpoint-1]+sorted_xs[hi_midpoint])/2

def median(v:List[float])->float:
    """v의 중앙값을 계산"""
    return _median_even(v) if len(v)%2==0 else _median_odd(v)

#분위
def quantile(xs:List[float],p:float)->float:
    """x의 p분위에 속하는 값을 반환"""
    p_index=int(p*len(xs))
    return sorted(xs)[p_index]