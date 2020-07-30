#분할할 문자열
test_sentence="this is a test sentence"

#split으로 리스트를 만듭니다.
split=test_sentence.split(" ")
print("split:",split)
#나눌 문자열.split("구분기호",분할할 횟수)

# 문제1
self_data="My name is Yuri"
#self_data를 분할해서 리스트 작성
#이름 부분 출력
answer=self_data.split(" ")
answer=answer[3]
print(answer)


#리스트 분할(여러 기호로 분할)
#표준 split()함수에서는 한 번에 여러 기호로 문자열을 분할할 수 없습니다. 
#여러 기호로 문자열을 분할할 경우에는 re모듈에 포함된 re.split()함수를 사용
import re
test_sentence="this,is a.test,sentence"
#",", " "와 "." 으로 분할해서 리스트를 만듭니다.
answer=re.split("[, .]",test_sentence)
print(answer)
#re.split()함수의 사용법
#re.split("[구분기호]", 분할할 문자열)
time_data="2020/1/5_22:15"
time_list=re.split("[/_:]",time_data)
print(time_list[1])
print(time_list[3])