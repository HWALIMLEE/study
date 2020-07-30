"""
map vs list
>>map은 본래 반복자 작성에 특화, list()함수로 배열을 생성할 때 계산 시간 늘어남
  따라서 map과 같은 방법으로 단순히 배열을 생성하려면 for루프의 리스트 내포 사용

"""
a=[1,-2,3,-4,5]
print([abs(x)for x in a])

print(list(map(abs,a)))

#반복자를 만들 때는 map을 사용하고, 배열을 만들 때는 리스트 내포를 사용하는 것이 좋다

minute_data=[30,155,180,74,11,60,82]

#분을 [시,분]으로 변환하는 함수 작성

func=lambda x:[x//60, x % 60]
print([func(x) for x in minute_data])


