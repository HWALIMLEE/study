#assert는 지정된 조건이 충족되지 않는다면 AssertionError반환
assert 1+1==2
assert 1+1==2, "1+1 should equal 2 but didn't" #조건이 충족되지 않을 때 출력하고 싶은 문구 추가

#코드를 작성할 때 꼭 assert를 사용해서 테스트 해보기
def smallest_item(xs):
    return min(xs)
assert smallest_item([10,20,5,40])==5
assert smallest_item([1,0,-1,2])==-1
# 이 구문은 예상하지 않은 프로그램의 상태를 확인하기 위해 활용해야 합니다. 
# 구문의 조건을 만족하지 않으면 프로그램이 정상적으로 실행되지 않고 종료되는데, 이는 프로그램의 버그가 있다는 것을 의미합니다.

def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)
    #대부분의 경우 코드가 제대로 작성되었는지 테스팅을 할 때 assert사용
    

