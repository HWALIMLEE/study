#파이썬은 동적 타입 언어-변수 올바르게만 사용하면 변수의 타입은 신경 쓰지 않아도 된다. 

def add(a,b):
    return a+b
assert add(10,5)==15,
assert add([1,2],[3])==[1,2,3],
assert add("hi","there")=="hi there",

try:
    add(10,"five")
except TypeError:
    print("cannot add an int to a string")

#mypy처럼 코드를 실행하기 전에 코드를 불러와서 타입 관련된 에러를 검사해 주는 도구
#mypy로 add("hi","there")가 포함되어 있는 파일을 검사하면 다음과 같은 에러 출력
#error:Argument 1 to "add" has incompatible type "str"; expected "int"


#타입 어노테이션 하는 방법
def total(xs:list)->float:
    return sum(total)

from typing import List
def total(xs:List[float])->float:
    return sum(total)

from typing import Callable
def twice(repeater:Callable[[str,int],str],s:str)->str:
    return repeater(s,2)
def comma_repeater(s:str,n:int)->str:
    n_copies=[s for _ in range(n)]
    return ','.join(n_copies)