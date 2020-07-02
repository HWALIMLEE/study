import p11_car
import p12_tv

print("=============================")
print("do.py의 module이름은", __name__)
print("=============================")

p11_car.drive()
p12_tv.watch()

# 운전하다
# car.py의 module 이름은 p11_car(import한 것-파일명)
# 시청하다
# tv.py의 module 이름은 p12_tv (import한 것-파일명)
# =============================
# do.py의 module이름은 __main__(실행시킨 시점)
# =============================
# 운전하다
# 시청하다