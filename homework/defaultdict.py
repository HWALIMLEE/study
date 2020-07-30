#defaultdict()는 딕셔너리를 만드는  dict클래스의 서브 클래스
#defaultdict()는 인자로 주어진 객체의 기본값을 딕셔너리값의 초기값으로 지정할 수 있다datetime A combination of a date and a time. Attributes: ()

from collections import defaultdict
int_dict=defaultdict(int)
print("ind_dict:",int_dict) #int
print(int_dict['key1']) #값을 지정하지 않은 키는 그 값이 0으로 지정
int_dict['key2']='test'
print("int_dict:",int_dict)

#defaultdict라는 말 그대로 처음 키를 지정할 떄 값을 주지 않으면 해당 키에 대한 값을 디폴트 값으로 지정하겠다
list_dict=defaultdict(list)
print("list_dict:",list_dict) #list
print(list_dict['key1']) #[]
list_dict['key2']='test'
print("list_dict:",list_dict)


#defaultdict()를 언제 사용하는 것이 좋을까?
#키의 개수를 세어야 하는 상황이나, 리스트나 셋의 항목을 정리해야 하는 상황에 적절
#default(int)활용예제
#문자열에 나타난 알파벳의 횟수를 계산하는 방법
letters='dongdongfather'
letters_dict=defaultdict(int)
for k in letters:
    letters_dict[k]+=1
print(letters_dict)

#defaultdict()를 사용하지 않을 때
letters='dongdongfather'
letters_dict={}
for k in letters:
    if not k in letters_dict: #키가 있는 지 확인
        letters_dict[k]=0 #없으면 0으로 초깃값 할당
    letters_dict[k]+=1

print(letters_dict)

#defaultdict(list)활용예제
name_list=[('kim','sungsu'),('kang','hodong'),('park','jisung'),('kim','yuna'),('park','chanho'),('kang','hodong')]
ndict=defaultdict(list)
for k,v in name_list: #리스트의 요소가 튜플이기 때문에 k,v값으로 할당
    ndict[k].append(v)
print(ndict)

#kang hodong이 두번 중복
#중복자료 없애줄 때, set 사용
#defaultdict(set)예제
#set은 파이썬의 데이터 구조 중 유일한 항목의 집합을 나타내는 구조
name_list=[('kim','sungsu'),('kang','hodong'),('park','jisung'),('kim','yuna'),('park','chanho'),('kang','hodong')]
nset=defaultdict(set)
for k,v in name_list:
    nset[k].add(v)  #set 은 append사용하지 못함
print(nset)
