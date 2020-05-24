#가설을 바라보는 또 다른 관점 p-value
#H0가 참이라고 가정하고, 실제로 관측된 값보다 더 극단적인 값이 나올 확률을 구하는 것
"""
from scratch.probability import normal_cdf
def normal_probability_above(lo:float,mu:float, sigma:float=1)->float:
    return 1-normal_cdf(lo,mu,sigma)
normal_probability_below=normal_cdf
def two_sided_p_value(x:float, mu:float=0, sigma:float=1)->float:

    mu(평균)와 sigma(표준편차)를 따르는 정규분포에서 x같이 
    극단적인 값이 나올 확률은 얼마일까?
    
    if x>=mu:
        #만약 x가 평균보다 크다면 x보다 큰 부분이 꼬리이다.
        return 2*normal_probability_above(x,mu,sigma)
    else:
        return 2*normal_probability_below(x,mu,sigma)
""" 

#시물레이션
import random
extreme_value_count=0
for _ in range(1000):
    num_heads=sum(1 if random.random()<0.5 else 0 for _ in range(1000))
    if num_heads>=530 or num_heads<=470:
        extreme_value_count+=1
print(extreme_value_count)
    