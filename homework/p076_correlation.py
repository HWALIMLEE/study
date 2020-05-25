#공분산-두 변수가 각각의 평균에서 얼마나 멀리 떨어져 있는 지 살펴본다.
#공분산은 단위의 문제가 발생
#따라서 공분산에서 각각의 표준편차를 나눠 준 상관관계를 더 자주 살펴본다.

def correlation(xs:List[float],ys:List[float])->float:
    stdev_x=standard_deviation(xs)
    stedv_y=standard_deviation(ys)
    if stdev_x>0 and stdev_y>0:
        return covariance(xs, ys)/stdev_x/stdev_y
    else:
        return 0 #편차가 존재하지 않는다면 상관관계는 0

#상관관계는 단위의 영향을 없앤 것
#-1<=correlation<=1

#상관관계와 인과관계는 같을 수도 있고 다를 수도 있다.
#두 변수가 상관이 있다고 해서 그것이 인과적으로 영향을 미치는 것은 아니다. 


