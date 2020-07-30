# 정규 표현식은 특정한 규칙을 가진 문자열의 패턴 표현하는데 사용하는 표현식
# 텍스트에서 특정 문자열을 검색하거나 치환할 때 흔히 사용

# 1. 특정 문자열을 직접 리터럴로 사용하여 해당 문자열 검색하는 것
import re
text = "에러 1122: 레퍼런스 오류\n 에러 1033: 아규먼트 오류"
regex = re.compile('에러 1033')
mo = regex.search(text)
if mo!=None:
    print(mo.group()) # 에러 1033

# 2. 특정 패턴의 문자열을 검색
text = "문의 사항이 있으면 032-232-3245으로 연락주시기 바랍니다."
regex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
matchobj = regex.search(text)
phonenumber = matchobj.group()
print(phonenumber) # 032-232-3245 

# 3. 다양한 정규식 패턴 표현
text = "에러 1122: 래퍼런스 오류\n 에러 1033: 아규먼트 오류"
regex = re.compile("에러\s\d+") # \s : 공백, \d+:숫자가 하나 이상 
mc = regex.findall(text)
print(mc)

# 4. 정규식 그룹
text = "문의사항이 있으면 032-232-3245으로 연락주시기 바랍니다."
regex = re.compile(r'(\d{3})-(\d{3}-\d{4})')
matchobj = regex.search(text)
areaCode = matchobj.group(1)
num = matchobj.group(2)
fullNum = matchobj.group()
print(areaCode, num) #032 232-3245
print(fullNum) #032-232-3245

# 5. Named Capturing Group을 사용하는 방법===>(?P<그룹명>정규식)
regex = re.compile(r'(?P<area>\d{3})-(?P<num>\d{3}-\d{4})')
matchobj = regex.search(text)
areaCode = matchobj.group("area")
num = matchobj.group("num")
print(areaCode, num) # 032 232-3245
