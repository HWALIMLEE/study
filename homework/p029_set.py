#집합 (set)은 파이썬의 데이터 구조 중 유일한 항목의 집합을 나타내는 구조
#집합은 중괄호 사용해서 정의
primes_below_10={2,3,5,7}

s=set()
s.add(1) #{1}
s.add(2) #{1,2}
s.add(2) #{1,2} 유일한 항목만
x=len(s) #2
y=2 in s #True
z=3 in s #False

#in은 집합에서 굉장히 빠르게 작동

#수많은 항목 중에서 특정 항목의 존재 여부를 확인해보기 위해서는 리스트를 사용하는 것보다 집합을 사용하는 것이 훨씬 효율적이다. 

#두번째 이유는 중복된 원소 제거해 주기 때문
item_list=[1,2,3,1,2,3]
num_items=len(item_list)
item_set=set(item_list)
num_distinct_items=len(item_set)
distinct_item_list=list(item_set)

print(item_set)
print(distinct_item_list)
#그러나 집합보다 딕셔너리나 리스트를 더 자주 사용할 것