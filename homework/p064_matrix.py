#행렬은 2차원으로 구성된 숫자의 집합
#타입 명시를 위한 별칭
Matrix=List[List[float]]

A=[[1,2,3],
[4,5,6]]

B=[[1,2],
[3,4],
[5,6]]

#행렬을 리스트의 리스트로 나타내는 경우, 행렬 A는 len(A)개의 행과 len(A[0])개의 열로 구성

from typing import Tuple
def shape(A:Matrix)->Tuple[int,int]:
    """(열의 개수, 행의 개수)를 반환"""
    num_rows=len(A)
    num_cols=len(A[0]) if A else 0
    return num_rows, num_cols

assert shape([[1,2,3],[4,5,6]])==(2,3)

def get_row(A:Matrix, i:int)->Vector:
    """A의 i번째 행을 반환"""
    return A[i]


def get_column(A:Matrix, j:int)->Vector:
    """A의 j번째 열을 반환"""
    return [A_i[j] for A_i in A] #각 A_i의 행에 대해

from typing import Callable
def make_matrix(num_rows:int,
num_cols:int,
entry_fn:Callable[[int,int],float])->Matrix:
    return [[entry_fn(i,j) for j in range(num_cols)]for i in range(num_rows)]

#단위 행렬 반환
def identity_matrix(n:int)->Matrix:
    return make_matrix(n,n,lambda i,j:1 if i==j else 0)

