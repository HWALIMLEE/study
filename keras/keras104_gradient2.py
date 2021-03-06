import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2-4*x +6
# x = np.linspace(-1,6,100) # -1부터 6까지 100간격
# y = f(x)

# # 그리자
# plt.plot(x,y,'k-')
# plt.plot(2,2,'sk')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


gradient = lambda x: 2*x-4 # f 함수 미분
# gradient가 0이 되는 부분이 최소의 loss값임

x0 = 0.0
MaxIter = 10
learning_rate = 0.25


print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1
    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))

"""
learning_rate = 0.25
step    x       f(x)
00      0.00000 6.00000
01      1.00000 3.00000
02      1.50000 2.25000
03      1.75000 2.06250
04      1.87500 2.01562
05      1.93750 2.00391
06      1.96875 2.00098
07      1.98438 2.00024
08      1.99219 2.00006
09      1.99609 2.00002
10      1.99805 2.00000

learning_rate = 0.5
step    x       f(x)
00      0.00000 6.00000
01      2.00000 2.00000
02      2.00000 2.00000
03      2.00000 2.00000
04      2.00000 2.00000
05      2.00000 2.00000
06      2.00000 2.00000
07      2.00000 2.00000
08      2.00000 2.00000
09      2.00000 2.00000
10      2.00000 2.00000

learning_rate = 0.1
step    x       f(x)
00      0.00000 6.00000
01      0.40000 4.56000
02      0.72000 3.63840
03      0.97600 3.04858
04      1.18080 2.67109
05      1.34464 2.42950
06      1.47571 2.27488
07      1.58057 2.17592
08      1.66446 2.11259
09      1.73156 2.07206
10      1.78525 2.04612

learning_rate = 0.2
step    x       f(x)
00      0.00000 6.00000
01      0.80000 3.44000
02      1.28000 2.51840
03      1.56800 2.18662
04      1.74080 2.06718
05      1.84448 2.02419
06      1.90669 2.00871
07      1.94401 2.00313
08      1.96641 2.00113
09      1.97984 2.00041
10      1.98791 2.00015


learning_rate = 0.25, MaxIter = 15
step    x       f(x)      
00      0.00000 6.00000   
01      1.00000 3.00000   
02      1.50000 2.25000   
03      1.75000 2.06250   
04      1.87500 2.01562   
05      1.93750 2.00391   
06      1.96875 2.00098   
07      1.98438 2.00024   
08      1.99219 2.00006   
09      1.99609 2.00002
10      1.99805 2.00000
11      1.99902 2.00000
12      1.99951 2.00000
13      1.99976 2.00000
14      1.99988 2.00000
15      1.99994 2.00000


learning_rate = 0.1, MaxIter = 20
00      0.00000 6.00000
01      0.40000 4.56000
02      0.72000 3.63840
03      0.97600 3.04858
04      1.18080 2.67109
05      1.34464 2.42950
06      1.47571 2.27488
07      1.58057 2.17592
08      1.66446 2.11259
09      1.73156 2.07206
10      1.78525 2.04612
11      1.82820 2.02951
12      1.86256 2.01889
13      1.89005 2.01209
14      1.91204 2.00774
15      1.92963 2.00495
16      1.94371 2.00317
17      1.95496 2.00203
18      1.96397 2.00130
19      1.97118 2.00083
20      1.97694 2.00053
"""