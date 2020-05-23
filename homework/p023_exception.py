#코드가 뭔가 잘못됐을 때 파이썬은 예외가 발생했음을 알려줌
try:
    print(0/0)
except ZeroDivisionError:
    print("cannot divide by zero")

