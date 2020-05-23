#정규표현식을 사용하면 문자열을 찾을 수 있다.
import re
re_examples=[
    not re.match("a","cat"), #문자열의 시작이 정규표현식과 같은 지 비교
    re.search("a","cat"), #문자열 전체에서 정규표현식과 같은 부분이 있는지 찾기
    not re.search("c","dog")
    3==len(re.split("[ab]","carbs")),

    "R-D-"==re.sub("[0-9]","-","R2D2")]

assert all(re_examples), "all the reges examples shoule be True"