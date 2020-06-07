#p116
fruits_name_1="사과"
fruits_num_1=2
fruits_name_2="귤"
fruits_num_2=10

fruits=[[fruits_name_1,fruits_num_1],[fruits_name_2,fruits_num_2]]

print(fruits)
print("--------------------------------------")
#p123
alphabet=["a","b","c"]
alphabet_copy=alphabet
alphabet_copy[0]='A'
print([alphabet])
print("--------------------------------------")

#p132
x=5
while x!=0:
    x-=1
    if x!=0:
        print(x)
    else:
        print("Bang")
print("--------------------------------------")
#p133
animals=["tiger","dog","elephant"]
for animal in animals:
    print(animal)
print("------------------------------------")
#p134
storages=[1,2,3,4,5,6,7,8,9,10]
for n in storages:
    print(n)
    if n>=5:
        print("끝")
        break
print("-----------------------------------")
for n in storages:
    print(n)
    if n==4:
        break
print("----------------------------------")
#p136
storages=[1,2,3]
for n in storages:
    if n==2:
        continue
    print(n)
#continue는 break와 마찬가지로 if 문 등과 조합해서 사용하지만 break와 달리 특정 조건일 때 루프를 한 번 건너뜁니다.
print("------------------------------------------")

storages=[1,2,3,4,5,6]
for n in storages:
    if n%2==0:
        continue
    print(n)
print("--------------------------------------------")

#137
#for문에서 index표시
#enumerate()함수를 사용하면 인덱스가 포함된 요소를 얻을 수 있습니다
list=['a','b']
for index, value in enumerate(list):
    print(index,value)

print("--------------------------------------------")
animals=['tiger','dog','elephant']
for index,value in enumerate(animals):
    print("index"+str(index),value)

print("-----------------------------------------------")

#138
#리스트 안의 리스트 루프
list=[[1,2,3],
        [4,5,6]]
for a,b,c in list:
    print(a,b,c)

fruits=[['strawberry','red'],
        ['peach','pink'],
        ['banana','yellow']]

for fruit, color in fruits:
    print(fruit,'',"is",'',color)

print("------------------------------------------")

#140
#딕셔너리형의 루프-->.items()
fruits={'strawberry':'red','peach':'pink','banana':'yellow'}
for fruit, color in fruits.items():
    print(fruit+" is "+color)

town={"경기도":"분당","서울":"중구","제주도":"제주시"}

for city,town in town.items():
    print(city+" "+town)

#연습문제
items={"지우개":[100,2],"팬":[200,3],"노트":[400,5]}
total_price=0
for item in items:
    print(item+" 은(는) 한 개에 " + str(items[item][0])+"원이며,"+str(items[item][1])+"개 구입합니다.")

for item in items:
    total_price=int(items[item][0])*int(items[item][1])

    

print("지불해야 할 금액은 "+ total_price +"원 입니다.")

money=()
if money>total_price:
    print("거스름돈은 "+(money-total_price)+"원 입니다.")
elif money==total_price:
    print("거스름돈은 없습니다.")
else:
    print("돈이 부족합니다.")