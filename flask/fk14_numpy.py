import pymssql as ms
import numpy as np

conn = ms.connect(server='127.0.0.1',user='bit2', password='1234', 
                  database='bitdb')
cursor = conn.cursor()

cursor.execute("SELECT * FROM iris2;")

row = cursor.fetchall()
print(row)
conn.close()

print("=========================================")
aaa = np.array(row)
print(aaa)
print(aaa.shape) # (150, 5)
print(type(aaa)) 
 
np.save("./data/test_flask_iris2.npy",aaa) # 제주 데이터 써먹기/ 큰 파일은 모두 db에 저장


