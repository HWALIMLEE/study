from collections import Counter
from matplotlib import pyplot as plt
grades=[83,95,91,87,70,0,85,82,100,67,73,77,0]
#점수는 10점 단위로 그룹화
histogram=Counter(min(grade//10*10,90)for grade in grades) #//--->소수점 이하 버린다. 
plt.bar([x+5 for x in histogram.keys()], #각 막대를 오른쪽으로 5만큼 옮기고
histogram.values(), #각 막대의 높이를 정해 주고
10, #높이는 10
edgecolor=(0,0,0))
plt.axis([-5,105,0,5]) 
plt.xticks([10*i for i in range(11)])
plt.xlabel("Decile")
plt.ylabel("#of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()

#y축 조심
mentions=[500,505]
years=[2017,2018]

plt.bar(years,mentions,0.8)
plt.xticks(years)
plt.ylabel("#of times I hears someone say 'data_science'")

plt.ticklabel_format(useOffset=False)
plt.axis([2016.5,2018.5,0,550])
plt.title("Not So Huge Anymore!")
plt.show()
