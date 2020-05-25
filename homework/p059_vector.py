#벡터는 float객체를 갖고 있는 리스트인 Vector라는 타입으로 명시
from typing import List
Vector=List[float]
height_weight_age=[70,170,40]

#백터 덧셈--->zip 사용
def add(v:Vector,w:Vector)->Vector:
    """각 성분끼리 더한다"""
    assert len(v)==len(w), "vectors must be the same length"
    return [v_i+w_i for v_i, w_i in zip(v,w)]

assert add([1,2,3],[4,5,6])==[5,7,9]

def subtract(v:Vector,w:Vector)->Vector:
    """각 성분끼리 뺀다."""
    assert len(v)==len(w), "vectors must be the same lentgh"
    return [v_i-w_i for v_i, w_i in zip(v,w)]
assert subtract([5,7,9],[4,5,6])==[1,2,3]

#벡터로 구성된 리스트에서 모든 벡터의 각 성분 더하기
def vector_sum(vectors:List[Vector])->Vector:
    """모든 벡터의 각 성분들끼리 더한다."""
    #vectors가 비어있는 지 확인
    assert vectors, "no vectors provided!"

    #모든 벡터의 길이가 동일한 지 확인
    num_elements=len(vectors[0])
    assert all(len(v)==num_elements for v in vectors), "different sizes!"
    #i번째 결과값은 모든 벡터의 i번째 성분을 더한 값
    return [sum(vector[i]for vector in vectors)
    for i in range(num_elements)]

assert vector_sum([[1,2],[3,4],[5,6],[7,8]])==[16,20]

#벡터에 스칼라 곱해줄 수 있어야 한다.
def scalar_multiply(c:float, v:vector)->Vector:
    """모든 성분을 c로 곱하기"""
    return [c*v_i for v_i in v]
assert scalar_multiply(2,[1,2,3])==[2,4,6]

def vector_mean(vectors:List[Vector])->Vector:
    """각 성분별 평균을 계산"""
    n=len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1,2],[3,4],[5,6]])==[3,4]

#벡터의 내적은 벡터의 각 성분별 곱한 값을 더해준 값
#내적은 v가 w로 투영된 벡터의 길이를 나타낸다. 
def dot(v:Vector, w:Vector)->float:
    """v_1*w_1+...+v_n*w_n"""
    assert len(v)==len(w), "vectors must be same length"

    return sum(v_i*w_i for v_i, w_i in zip(v,w))

assert dot([1,2,3],[4,5,6])==32

#내적의 개념을 사용하면 각 성분의 제곱 값의 합을 쉽게 구할 수 있다. 
def sum_of_squares(v:Vector)->float:
    """v_1*v_1+...v_n*v_n"""
    return dot(v,v)

assert sum_of_squares([1,2,3])==14

#제곱 값의 합을 이용하면 벡터의 크기 계산
import math

def magnitude(v:Vector)->float:
    """벡터 v의 크기를 반환"""
    return math.sqrt(sum_of_squares(v))
assert magnitude([3,4])==5

#두 벡터간 거리 구하기
def squared_distance(v:Vector, w:Vector)->float:
    return sum_of_squares(subtract(v,w))

def distance(v:Vector, w:Vector)->float:
    return math.sqrt(squared_distance(v,w))
#수정
def distance(v:Vector, w:Vector)->float:
    return magnitude(subtract(v,w))
