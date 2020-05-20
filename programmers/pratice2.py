def solution(phone_book):
    phone_book.sort()
    i=0
    while i<(len(phone_book))-1:
        for p in phone_book[i+1:]:
            con = not(p.startswith(p[i]))
            if con==False:
                return con
            if con==True:
                i+=1
    return True
    

    #오류 발생
    