#기존의 리스트에서 특정 항목을 선택하거나 변환시킨 결과를 새로운 리스트에 저장해야 하는 경우
even_numbers=[x for x in range(5)if x %2==0] #[0,2,4]
squares=[x*x for x in range(5)] #[0,1,4,9,16]
even_squares=[x*x for x in even_numbers] #[0,4,16]

#딕셔너리 변환
square_dict={x:x*x for x in range(5)} #{0:0,1:1,2:4,3:9,4:16}
square_set={x*x for x in [1,-1]} #{1}

#리스트에서 불필요한 값은 밑줄
zeros=[0 for _ in even_numbers] #[0,0,0] --->even_numbers와 동일한 길이 [0,2,4]대신 [0,0,0]을 넣어라
print(zeros)

pairs=[(x,y)
for x in range(10)
for y in range(10)]
print("pairs:",pairs) #(0,0),(0,1).....(9,9)까지

increasing_pairs=[(x,y)
for x in range(10)
for y in range(x+1,10)]
print("increasing_pairs:",increasing_pairs) #(0,1),(0,2)....(0,9),(1,2).......(8,9)

