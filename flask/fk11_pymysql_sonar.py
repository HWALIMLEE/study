import pymssql as ms
# print("continue")

conn = ms.connect(server = 'localhost', user = 'bit2', 
                    password='1234', database = 'bitdb')

print("끗")

#연결
cursor = conn.cursor()

#데이터 불러오기
cursor.execute("SELECT * FROM sonar;")

# 한 줄 가져오기
row = cursor.fetchone() 

while row:
    print("첫칼럼: %s, 두번째 컬럼: %s" %(row[0], row[1]))
    row = cursor.fetchone()

# connection된 거는 마지막에 끊어주기
conn.close()




