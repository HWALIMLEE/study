#딕셔너리는 파이썬의 또 다른 기본적 데이터 구조, key:value값
empty_dict={}
empty_dict2=dict()
grades={"Joel":80, "Tim":95}

joels_grade=grades["Joel"]
print(joels_grade)

#딕셔너리에 존재하지 않는 키
try:
    kates_grade=grades["Kate"]
except KeyError:
    print("no grade for Kate!")

#연산자 in 
joel_has_grade="Joel" in grades
kate_has_grade="Kate" in grades

print(joel_has_grade)
print(kate_has_grade)

#크기가 큰 딕셔너리에서도 키의 존재 여부 빠르게 확인
joels_grade=grades.get("Joel",0)  #key값이 존재하면 value값 반환/ 
kates_grade=grades.get("Kate",0) #key값이 존재하지 않으면 0반환/ 0을 쓰지 않으면 None반환
no_ones_grade=grades.get("No one")
print(joels_grade)
print(kates_grade)
print(no_ones_grade)

grades["Tim"]=99 #기존 값 대체(95-->99)
grades["Kate"]=100 #Kate:100추가
num_students=len(grades) #총 3명이 됨

#딕셔너리의 키는 수정할 수 없으며 리스트를 키로 사용할 수 없다. 
tweet={
    "user":"joelgrus","text":"Data Science is Awesome","retweet_count":100,"hashtags":["#data","#science","#datascience","#awesome","#yolo"]
}
print(tweet)
tweet_keys=tweet.keys()
tweet_values=tweet.values()
tweet_items=tweet.items()

print(tweet_keys) #키에대한 리스트
print(tweet_values) #값에 대한 리스트
print(tweet_items) #(key,value) 튜플에 대한 리스트

