class CountingClicker:
    def __init__(self,count=0):
        self.count=count

clicker1=CountingClicker()
clicker2=CountingClicker(100)
clicker3=CountingClicker(count=100)

print(clicker1)
print(clicker2)
print(clicker3)

def __repr__(self):
    return f"CountingClicker(count={self.count})"

def click(self,num_times=1):
    """한 번 실행할 때마다 num_times만큼 count증가"""
    self.count+=num_times

def read(self):
    return self.count

def reset(self):
    self.count=0

clicker=CountingClicker()
assert clicker.read()==0, "clicker should start with count 0"
clicker.click()
clicker.click()
assert clicker.read()==2, "after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read()==0, "after reset, clicker should be back to 0"

#부모 클래스의 모든 기능을 상속받는 서브클래스
class NoResetClicker(CountingClicker):
    def reset(self):
        pass

clicker2=NoResetClicker()
assert clicker2.read()==0
clicker2.click()
assert clicker2.read()==1
clicker2.reset()
assert clicker2.read()==1, "reset shouldn't do anything"

