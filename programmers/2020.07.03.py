#2016년
def solution(a, b):
    import datetime
    answer = ''
    w = ['MON','TUE','WED','THU','FRI','SAT','SUN']
    answer = w[datetime.date(2016,a,b).weekday()]
    return answer

# 체육복
