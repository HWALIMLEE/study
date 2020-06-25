from machine.car import drive
from machine.tv import watch

drive()
watch()

from machine import car
from machine import tv

print("=======================")
car.drive()
tv.watch()

print("=========================")

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()
print("===========================")
from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

print("============================")
from machine import test
from machine import tv

test.car.drive()
tv.watch()

# 같은 폴더에 하위폴더 있으면 얼마든지 불러올 수 있음
