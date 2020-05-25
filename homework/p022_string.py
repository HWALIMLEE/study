<<<<<<< HEAD
not_tab_string=r"/t"
print(not_tab_string) #/t

#세 개의 따옴표 사용하여 하나의 문자열을 여러 줄로 나눠서 나타내기
multi_line_string="""This is the first line.
and this is the second line
and this is the third line"""    

print(multi_line_string)
=======
#개행문자 방지는 앞에 r붙이기
not_tab_string=r"/t" 
print(not_tab_string)

#세개 따옴표 사용 한 문자열을 여러 줄로 나눠서 출력
multi_line_string="""This is the first line.
and this is the second line
and this is the third line"""
print(multi_line_string)

#기존방식
first_name="Joel"
last_name="Grus"

full_name1=first_name+" "+last_name
print(full_name1)
full_name2="{0} {1}".format(first_name,last_name)
print(full_name2)

#f-string사용
full_name3=f"{first_name} {last_name}"
print(full_name3)

>>>>>>> ccc518f648cb90e9f1a38f798788c796b5835a47
