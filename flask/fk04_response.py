from flask import Flask
app = Flask(__name__)

from flask import make_response

@app.route('/')
def index():
    response  = make_response('<h1> 잘 따라 치시오!!! </h1>')
    response.set_cookie('answer','42')
    return response

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)

"""
cookie vs session

cookie: 쿠키는 클라이언트의 PC에 텍스트 파일 형태로 저장되는 것으로 일반적으로는 시간이 지나면 소멸한다.
보통 세션과 더불어 자동 로그인, 팝업 창에서 "오늘은 이 창을 더 이상 보지 않기" 등의 기능을 클라이언트에 저장해놓기 위해 사용된다.

session: 쿠키와 다르게 세션과 관련된 데이터는 서버에 저장된다. 
서버에서 관리할 수 있다는 점에서 안전성이 좋아서 보통 로그인 관련으로 사용되고 있다. 
플라스크에서 세션은 딕셔너리의 형태로 저장되며 키를 통해 해당 값을 불러올 수 있다.
"""

