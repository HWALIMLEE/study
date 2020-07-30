# from flask import Flask, render_template, request
# import sqlite3

# app = Flask(__name__)

# # 데이터 베이스
# conn = sqlite3.connect("./data/wanggun.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM general")
# print(cursor.fetchall())


# # 웹 상에서 적용 되는 것
# @app.route('/')
# def run():
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM general")
#     rows = c.fetchall();
#     return render_template("board_index.html",rows=rows)



# # 웹 상 수정
# # /modi
# @app.route('/modi')
# def modi():
#     ids = request.args.get('id')
#     conn = sqlite3.connect('./data/wanggun.db')
#     c = conn.cursor()
#     c.execute('SELECT * FROM general WHERE id ='+str(ids))   # id 를 설정해주면 설정한 id에 대한 값만 출력. 
#     rows = c.fetchall();
#     return render_template('board_modi.html', rows = rows)

    
    

# @app.route('/addrec',methods=['POST','GET'])
# def addrec():
#     if request.method == 'POST':
#         try:
#             war = request.form['war']
#             id1 = request.form['id']
#             with sqlite3.connect("./data/wanggun.db") as conn:
#                 cur = conn.cursor()
#                 cur.execute("UPDATE general SET war=" + str(war) + "WHERE id=" + str(id1))
#                 conn.commit()  # 수정하고 나면 항상 commit
#                 msg = "정상적으로 입력되었습니다."    
#         except:
#             conn.rollback()
#             msg = "입력과정에서 에러가 발생했습니다."
#         finally:
#             return render_template("board_result.html",msg = msg)
#             conn.close()
# app.run(host='127.0.0.1',port=5000,debug=False)

# 기태님 코드
from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# 데이터베이스 만들기
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general;")
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general;")
    rows = c.fetchall()
    return render_template("board_index.html", rows=rows)

@app.route('/modi')
def modi():
    ids = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general where id = ' + str(ids))
    rows = c.fetchall()
    return render_template('board_modi.html', rows=rows)

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            conn = sqlite3.connect('./data/wanggun.db')
            war = request.form['war']
            ids = request.form['id']
            c = conn.cursor()
            c.execute('UPDATE general SET war = '+ str(war) + " WHERE id = "+str(ids))
            conn.commit()
            msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '에러가 발생하였습니다.'
        finally:
            conn.close()
            return render_template("board_result.html", msg=msg)

app.run(host='127.0.0.1', port=5000, debug=False)

