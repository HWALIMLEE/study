import sqlite3 
# splite도 db(database)
# local로 깔려 있음
# 그냥 가져다 쓰면 된다
# sql은 삽입, 삭제, 수정 다 함
conn = sqlite3.connect("test.db") # 없으면 자동 생성 STUDY폴더에 자동 생성된다.(STUDY 폴더 하단에 설치됨)

cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT,
                FoodName TEXT, Company TEXT, price INTEGER)""")

sql = "DELETE FROM supermarket"  # 실행시킬때마다 지움, 항상 4개씩만 들어가게 됨
cursor.execute(sql)

# 데이터 넣자
sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (1,'과일','자몽','마트',1500)) #자료 순서대로 넣어주기

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (2,'음료수','망고주스','편의점',1000)) #자료 순서대로 넣어주기

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (3,'고기','소고기','하나로마트',10000)) #자료 순서대로 넣어주기

sql = "INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)"
cursor.execute(sql, (4,'박카스','약','약국',500)) #자료 순서대로 넣어주기


sql = "SELECT * FROM supermarket"
# sql = "SELECT Itemno,Category,FoodName,Company, Price From Supermarket"
cursor.execute(sql)
rows = cursor.fetchall()

for row in rows:
    print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
              str(row[3]) + " " + str(row[4]))
# 저장
conn.commit()
conn.close()
# 할때마다 저장되서 
# 처음에는 4개 그다음은 8개 그다음은 12개...
# 따라서 위에 이거 써주어야 한다 
# sql = "DELETE FROM supermarket" 
# cursor.execute(sql)

