#종속성과 독립성
#수학적으로 사건 E와 F가 동시에 발생할 확률이 각각 사건이 발생할 확률의 곱과 같다면 두 사건은 독립 사건을 의미한다. 
#P(E,F)=P(E)P(F)

#조건부 확률
#사건 F가 발생했을 경우, 사건 E가 발생ㅇ할 확률이라고 이해

import enum,random
#Enum을 사용하면 각 항목에 특정 값을 부여할 수 있으며
#파이썬 코드를 더욱 깔끔하게 만들어준다.
class Kid(enum.Enum):
    BOY=0
    GIRL=1

def random_kid()->Kid:
    return random.choice([Kid.BOY,Kid.GIRL]) #임의로 한명만 출력

both_girls=0
older_girl=0
either_girl=0

random.seed(0)


for _ in range(10000):
    younger=random_kid()
    older=random_kid()
    if older==Kid.GIRL:
        older_girl+=1
    if older==Kid.GIRL and younger==Kid.GIRL:
        both_girls+=1
    if older==Kid.GIRL or younger==Kid.GIRL:
        either_girl+=1
print("P(both|older:",both_girls/older_girl)
print("P(both|either):",both_girls/either_girl)