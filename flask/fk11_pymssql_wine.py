import pymssql as ms
# print("continue")

conn = ms.connect(server = 'localhost', user = 'bit2', 
                    password='1234', database = 'bitdb')

print("끗")

cursor = conn.cursor()

cursor.execute("SELECT * FROM wine;")

row = cursor.fetchone() # 한 줄 가져오기

while row:
    print("column1: %s, column2: %s" %(row[0], row[1]))
    row = cursor.fetchone()

# connection된 거는 마지막에 끊어주기
conn.close()




