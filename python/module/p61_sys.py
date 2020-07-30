import sys 
print(sys.path)

# 'C:\\Users\\bitcamp\\anaconda3\\python37.zip', 
# 'C:\\Users\\bitcamp\\anaconda3\\DLLs', 
# 'C:\\Users\\bitcamp\\anaconda3\\lib', 
# 'C:\\Users\\bitcamp\\anaconda3', 
# 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages', 
# 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\win32', 
# 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\win32\\lib', 
# 'C:\\Users\\bitcamp\\anaconda3\\lib\\site-packages\\Pythonwin']
# 상시적으로 쓰고 싶을 때
# 아무때나 불러올 수 있다
# 이 폴더 안에 아무데나 넣을 수 있다
# 환경변수 path안에도 넣을 수 있다. 

from test_import import p62_import
p62_import.sum2() #이 import는 아나콘다 폴더에 들어있다  # 파일을 import
                  # 작업그룹 임포트 썸탄다

from test_import.p62_import import sum2
sum2() #작업그룹 임포트 썸탄다(한문장만 출력) # 함수만 import 
