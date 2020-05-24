#Type Annotation이나 typing 패키지는 동적 타입인 파이썬을 정적 타입으로 만들어주지 않는다. 
#다만 변수나 함수 파라미터와 반환값이 어떤 타입인지 코드 상에서 명시할 수 있으며, 에디터 레벨에서 경고를 띄워줄 뿐이다.
from typing import Tuple
import math
def normal_approximation_to_binomial(n:int,p:float)->Tuple(float, float):
    """Binomial(n,p)에 해당되는 mu(평균)와 sigma(표준편차)계산"""
    mu=p*n
    sigma=math.sqrt(p*(1-p)*n)
    return mu,sigma

#확률변수가 정규분포를 따른다는 가정하에 normal_cdf를 사용하면 실제 동전 던지기로부터 얻은 값이 구간 안(혹은 밖)에 존재할 확률을 계산할 수 있다.
from scratch.probability import normal_cdf
#누적 분포 함수는 확률변수가 특정 값보다 작을 확률을 나타낸다. 
normal_probability_below=normal_cdf
#만약 확률변수가 특정 값보다 작지 않다면, 특정 값보다 크다는 것을 의미한다. 
def normal_probability_above(lo:float,
                            mu:float=0,
                            sigma:float=1)->float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo보다 클 확률 """"
    return 1-normal_cdf(lo,mu,sigma)
#만약 확률변수가 hi보다 작고, lo보다 작지 않다면 확률변수는 hi와 lo사이에 존재한다.

def normal_probability_between(lo:float, hi:float, mu:float=0,sigma:float=1)->float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 있을 확률"""
    return normal_cdf(hi,mu,sigma)-normal_cdf(lo,mu,sigma)

#만약 확률변수가 범위 밖에 존재한다면 범위 안에 존재하지 않다는 것을 의미
def normal_probability_outside(lo:float, hi:float, mu:float=0, sigma:float=1)->float:
    """mu(평균)와 sigma(표준편차)를 따르는 정규분포가 lo와 hi 사이에 있을 확률"""
    return 1-normal_probability_between(lo,hi,mu,sigma)

#평균을 중심으로 대칭적인 구간 구하기
from scratch.probability import inverse_normal_cdf
def normal_upper_bound(probability:float, mu:float=0, sigma:float=1)->float:
    """P(Z<=z)=probability인 z값을 반환"""
    return inverse_normal_cdf(probability,mu,sigma)
def normal_lower_bound(probability:float, mu:float=0, sigma:float=1)->float:
    """P(Z>=z)=probability인 z값을 반환"""
    return inverse_normal_cdf(1-probability,mu,sigma)
def normal_two_sided_bounds(probability:float, mu:float=0, sigma:float=1)->Tuple[float, float]:
    """입력한 probability값을 포함하고, 
    평균을 중심으로 대칭적인 구간을 반환"""
    tail_probability=(1-probability)/2
    #구간의 상한은 tail_probability값 이상의 확률을 갖고 있다.
    upper_bound=normal_lower_bound(tail_probability,mu,sigma)
    #구간의 하한은 tail_probability값 이하의 확률 값을 갖고 있다.
    lower_bound=normal_upper_bound(tail_probability,mu,sigma)
    return lower_bound, upper_bound


