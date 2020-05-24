#경사하강법
#실수 벡터를 입력하면 실수 하나를 출력해 주는 함수 f
from scratch.linear_algebra import Vector,dot
def sum_of_squares(v:Vector)->float:
    """v에 속해 있는 항목들의 제곱합을 계산한다."""
    return dot(v,v)

#그래디언트는 함수가 가장 빠르게 증가할 수 있는 방향을 나타낸다.
#함수의 최댓값을 구하는 방법 중 하나는 임의의 시작점을 잡은 후, 그래디언트를 계산하고, 그래디언트의 방향으로 조금 이동하는 과정을 여러 번 반복

#그래디언트 계산
from typing import Callable
def difference_quotient(f:Callable[[float],float],x:float, h:float)->float:
    return (f(x+h)-f(x))/h

def square(x:float)->float:
    return x*x

def derivative(x:float)->float:
    return 2*x

#아주 작은 e값 대입해 미분값을 어림잡을 수 있다.
xs=range(-10,11)
actuals=[derivative(x)for x in xs]
estimates=[difference_quotient(square,x,h=0.001)for x in xs]

#두 계산식의 결과값이 거의 비슷함을 보여 주기 위한 그래프
import matplotlib.pyplot as plt
plt.title("Actual Derivatives vs. Estimates")
plt.plot(xs,actuals, 'rx',label='Actual')
plt.plot(xs,estimates,'b+',label='Estimate')
plt.legend(loc=9)
plt.show()