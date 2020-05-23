#3. 딕셔너리 #중복 x 
# {키 : 밸류}
# {key : value}


a={1:'hi',2:'hello'}
print(a)
print(a[1]) #hi

b={'hi':1,'hello':2}
print(b['hello']) #2

# 딕셔너리 요소 삭제
del a[1]
print(a)

del a[2]
print(a)

a={1:'a',1:'b',1:'c'} #키 값 중복 X
print(a) #{1:'c'}가장 마지막 요소만 나옴

b={1:'a',2:'a',3:'a'} # value 중복 O
print(b)

a={'name':'yun','phone':'010','birth':'0511'}
print(a.keys()) #'name','phone','birth'
print(a.values()) #'yun','010','0511'
print(type(a)) 
print(a.get('name')) #yun
print(a['name']) #yun
print(a.get('phone')) #010
print(a['phone']) #010


