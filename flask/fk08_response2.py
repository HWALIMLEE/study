from flask import Flask, Response, make_response
app = Flask(__name__)

@app.route("/")
def response_test():
    custom_response = Response("[★] Custome Response",200,
                        {"Program" : "Flask Web Application"})
    return make_response(custom_response)     

# 첫 번째 요청에만 응답
@app.before_first_request
def before_first_request():
    print("[1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.")
    print("이 서버는 개인 자산이니 건들지 말 것")
    print("곧 자료를 전송합니다.")

# 계속된 요청에도 응답
# @app.route()보다 먼저 반응
@app.before_request
def before_request():
    print("[2] 매 HTTP요청이 처리되기 전에 실행됩니다.")

# app.route()보다 뒤에 반응
@app.after_request
def after_request(response):
    print("[3] 매 HTTP요청이 처리되고 나서 실행됩니다.")
    return response
#예외처리
@ app.teardown_request
def teardown_request(exception):
    print("[4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다")

@app.teardown_appcontext
def teardodwn_appcontext(exception):
    print("[5] HTTP요청의 애플리케이션 컨텍스트가 종료될 떄 실행된다")
    

if __name__=='__main__':
    app.run(host='127.0.0.1')
# 플라스크 기본 포트 : 5000

# route가 돌아가기 전에 before_first_request가 먼저 돌아감
# before_firest_request가 밑밥작업
# 예륻 들어서 누가 물어보기 이전에 답을 먼저 해주는 것

