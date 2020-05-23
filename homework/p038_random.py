import random
random.seed(10) #매번 동일한 결과를 반환해 주는 설정

four_uniform_randoms=[random.random() for _ in range(4)]

print(four_uniform_randoms)

random.seed(10)
print(random.random())

#인자가 1개 혹은 2개인 random.randrange메서드를 사용하면 range()에 해당하는 구간 안에서 난수 생성
random.randrange(10)
random.randrange(3,6) 
print(random.randrange(10)) #[0,1,...9]에서 난수 생성
print(random.randrange(3,6)) #[3,4,5]에서 난수 생성

#random.shuffle은 리스트의 항목을 임의 순서로 재정렬
up_to_ten=[1,2,3,4,5,6,7,8,9,10]
random.shuffle(up_to_ten)
print(up_to_ten)

#random.choice메서드를 사용하면 리스트에서 임의의 항목을 하나 선택
my_best_friend=random.choice(['Alice',"Bob","Charlie"])
print(my_best_friend)


#random.sample을 사용하면 리스트에서 중복이 허용되지 않는 임의의 표본 리스트를 만들 수 있다.
lottery_numbers=range(60)
winning_numbers=random.sample(lottery_numbers,6)
print(winning_numbers)
#중복이 허용되는 임의의 표본 리스트를 만들고 싶다면 random.choice메서드를 여러번 사용
four_with_replacement=[random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)



