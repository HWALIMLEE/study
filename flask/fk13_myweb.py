import pyodbc as pyo

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server}; SERVER=' +server+
                    '; PORT=1433; DATABASE=' + database +
                    '; UID=' + username+
                    '; PWD=' + password)

cursor = conn.cursor()

tsql = "SELECT * FROM iris2;"

# 출력하는 부분-현재는 필요 없음
# with curser.execute(tsql):
#     row = curser.fetchone()

#     while row:
#         print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
#               str(row[3]) + " " + str(row[4]))
#         row = curser.fetchone()


# flask와 연결
from flask import Flask, render_template #html불러오기 위해
app = Flask(__name__)

@app.route("/sqltable")
def showsql():
    cursor.execute(tsql) # "SELECT * FROM iris2;" 실행하겠다
    return render_template("myweb.html", rows = cursor.fetchall())

if __name__=="__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
    
conn.close()



