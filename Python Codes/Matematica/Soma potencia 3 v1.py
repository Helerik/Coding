


def main():
    
    x = 0
    y = 0
    z = 0
    
    soma = (x**3) + (y**3) + (z**3)

    n = 60

    A = [[1,1,1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[1,-1,1],[1,1,-1],[1,-1,-1]]

    loop = 0
    while soma != n:
        print("loops: ",loop)
        y = 0
        z = 0
        x+=1
        for i in range (len(A)):
            print("loops: ",loop)
            soma = (x**3) + (y**3) + (z**3)
            if soma == n:
                break
            x = x*(A[i][0])
            y = y*(A[i][1])
            z = z*(A[i][2])
            loop+=1
            
        if soma != n:
            while y != x:
                print("loops: ",loop)
                if soma == n:
                    break
                z = 0
                y+=1
                for i in range (len(A)):
                    print("loops: ",loop)
                    soma = (x**3) + (y**3) + (z**3)
                    if soma == n:
                        break
                    x = x*(A[i][0])
                    y = y*(A[i][1])
                    z = z*(A[i][2])
                    loop+=1
                if soma != n:
                    while z != y:
                        print("loops: ",loop)
                        if soma == n:
                            break
                        z+=1
                        for i in range (len(A)):
                            print("loops: ",loop)
                            soma = (x**3) + (y**3) + (z**3)
                            if soma == n:
                                break
                            x = x*(A[i][0])
                            y = y*(A[i][1])
                            z = z*(A[i][2])
                            loop+=1
                        loop+=1
                loop+=1
        loop+=1
    print()                   
    print("x, y, z =", x, y, z, "para n =", n)
    print("x^3, y^3, z^3 =", x**3, y**3, z**3, "para n =", n)

    
main()
