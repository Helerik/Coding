class fraction:


    ##reduz a fracao para sua forma mais simplificada
    def _reduce(self, n,d, sign):
        p = abs(n)
        q = abs(d)
        while p % q != 0:
            pp = p
            qq = q
            p = qq
            q = pp % qq
        n = abs(n)//q * sign
        d = abs(d)//q
        return(n, d)

    #funcao de classe que retorna o numerador
    def getnumerador(self):
        return self._numerador
    #funcao de classe que retorna o denominador
    def getdenominador(self):
        return self._denominador
    
    def __init__(self, numerador, denominador):
        
        if (not isinstance(numerador, int)):
            raise (TypeError("O numerador deve ser inteiro, ou uma fracao")) #autoexplicativo
        if (not isinstance(denominador, int)):
            raise (TypeError("O denominador deve ser inteiro, ou uma fracao"))

        if isinstance(numerador, fraction) or isinstance(denominador, fraction):
            return numerador

        if denominador == 0:
            raise ZeroDivisionError("O denominador da fracao nao pode ser zero") #autoexplicativo

        if numerador == 0:
            self._numerador = 0
            self._denominador = 1

        #determina o sinal da fracao, baseado no valor do numerador e/ou numerador
        else:
            if (numerador < 0 and denominador >= 0) or (numerador >=0 and denominador <0):
                sign = -1 
            else:
                sign = 1
            (self._numerador, self._denominador) = self._reduce(numerador, denominador, sign)

    def __repr__(self):
        return str(self._numerador)+"/"+str(self._denominador) #representacao grafica da fracao como string

    def __eq__(self, direita): #funcao interna de igualdade (comparativa)
        esquerda = self
        if esquerda.getnumerador() == direita.getnumerador() and esquerda.getdenominador() == direita.getdenominador():
            return True
        else:
            return False

    def __ne__(self, direita): #funcao interna de desigualdade comparativa
        esquerda = self
        return not esquerda == direita

    def __lt__(self, direita): #funcao interna de menor-que comparativa
        esquerda = self
        return (esquerda.getnumerador() * direita.getdenominador()) < (esquerda.getdenominador() * direita.getnumerador())

    def __le__(self, direita): #funcao interna de menor ou igual-que comparativa
        esquerda = self
        return not direita < esquerda

    def __gt__(self, direita): #funcao interna de maior-que comparativa
        esquerda = self
        return direita < esquerda

    def __ge__(self, direita): #funcao interna de maior ou igual-que comparativa
        esquerda = self
        return not direita > esquerda

    def __add__(self, direita): #funcao interna de adicao ( a/b + c/d = (a*d + c*b)/(b*d) )
        esquerda = self
        num = esquerda.getnumerador() * direita.getdenominador() + direita.getnumerador() * esquerda.getdenominador()
        den = direita.getdenominador() * esquerda.getdenominador()
        return fraction(num,den)

    def __sub__(self, direita): #funcao interna de subtracao ( equivale a acima )
        esquerda = self
        num = esquerda.getnumerador() * direita.getdenominador() - direita.getnumerador() * esquerda.getdenominador()
        den = direita.getdenominador() * esquerda.getdenominador()
        return fraction(num,den)

    def __mul__(self, direita): #funcao interna de multiplicacao ( a/b * c/d = (a*c)/(b*d) )
        esquerda = self
        num = esquerda.getnumerador() * direita.getnumerador()
        den = esquerda.getdenominador() * direita.getdenominador()
        return fraction(num,den)

    def __truediv__(self, direita): #funcao interna de divisao normal ( (a/b)/(c/d) = (a*d)/(c*b) )
        esquerda = self
        num = esquerda.getnumerador() *  direita.getdenominador()
        den = esquerda.getdenominador() * direita.getnumerador()
        return fraction(num,den)

    def __abs__(self): #funcao interna que retorna o abs (i.e. modulo) do valor
        num = abs(self.getnumerador())
        den = abs(self.getdenominador())
        return fraction(num, den)

from sys import exit

def MontaSistema(Matriz, Igualdades):
    
    M = Matriz
    Ig = Igualdades

    sistema = []
    for i in range(len(M)):
        sistema.insert(i, [])
        for j in range(len(M[0])+1):
            sistema[i].insert(j, 0)
    for i in range(len(M)):
        for j in range(len(M[0])):
            sistema[i][j] = M[i][j]
        sistema[i][-1] = Ig[i][0]

    return sistema

def MatrizTransp (matriz):

    M = matriz #para facilitar, tomo M como a matriz
    
    lin = len(M) # refere a j (numero de elementos da linha)
    col = len(M[0]) # refere a i (numero de elementos da coluna)

    transp = [] #metodo para criacao de uma matriz vazia
    for i in range(col):
        transp.insert(i, [])

    #transposicao da matriz, linha por linha (linhas viram colunas)
    for j in range(lin):
        for i in range(col):
            transp[i].insert(j, M[j][i])

    return transp

def MultiplicaMatriz(matriz1, matriz2):

    M1 = matriz1 #para facilitar, usei notacao simplificada
    M2 = matriz2
    
    lin1 = len(M1)
    col1 = len(M1[0])
    lin2 = len(M2)
    col2 = len(M2[0])
    
    MXM = [] ## matriz multiplicada (pequena funcao para cria-la, vazia)
    for i in range(lin1):
            MXM.insert(i, [])
            for j in range(col2):
                MXM[i].insert(j, fraction(0,1))
    if col1 != lin2: #por definicao, eh impossivel multiplicar uma matriz a direita que tenha um numero de colunas diferente do numero de linhas
                     #de uma matriz a esquerda
        return exit("ERROR: Tried to multiply incompatible matrices; Verify matrices multiplication order or scale (i.e. mxn)")

    for i in range (lin1):
        for j in range (col2):
            for k in range (lin2):
                MXM[i][j] = (MXM[i][j]) + ((M1[i][k])*(M2[k][j])) #executa a operacao de multiplicacao de matrizes
   
    return MXM


def pivot(A,c):

    ID = []
    for i in range(len(A)):
        ID.append([])
        for j in range(len(A[i])):
            ID[i].append(fraction(0,1))
            if i == j:
                ID[i][j] = fraction(1,1)  
    for i in range(len(A)):
        maxlin = max(range(i, len(A)), key=lambda j: abs(A[j][i]))
        if i != maxlin:
            ID[i], ID[maxlin] = ID[maxlin], ID[i]
    if c == 1:
        A2 = A[:]
        P = pivot(A2,0)
        A2 = MultiplicaMatriz(P,A2)
        for i in range(len(A2)):        
        #zera coluna
            for j in range(i+1, len(A2)):
                if A2[i][i] == fraction(0,1):
                    P2 = pivot(A2,0)
                    P = MultiplicaMatriz(P2,P)
                    A2 = MultiplicaMatriz(P,A2)
                    
                temp = fraction(-1,1)*A2[j][i]/A2[i][i]
                                                                
                for k in range(i, len(A2), 1):
                    if i == k:
                        A2[j][k] = fraction(0,1)
                    else:
                        A2[j][k] = A2[j][k] + temp*A2[i][k]
        return P
    else:
        return ID

def flip(A):
    A2 = []
    A3 = []
    for i in range(len(A)):
        A3.append([])
        
    for i in range(-1,-len(A)-1,-1):
        A2.append(A[i])
    for i in range(len(A2)):
        for j in range(-1,-len(A2[i])-1,-1):
            A3[i].append(A2[i][j])
            
    return A3
    

def luDecompose(A):              

    P = pivot(A,1)
    A = MultiplicaMatriz(P,A)
    L = [] #lower matrix
    for i in range(len(A)):
        L.insert(i, [])
        for j in range(len(A[0])):
            if j == i:
                L[i].append(fraction(1,1))
            else:
                L[i].append(fraction(0,1))           
    for i in range(len(A)):        
        #zera coluna
        for j in range(i+1, len(A)):
            temp = fraction(-1,1)*A[j][i]/A[i][i] #utiliza o pivo para zerar a coluna abaixo (no processo, altera os valores do A
                                                              #e, dependendo, do resultado, para encontrar o vetor.
            for k in range(i, len(A), 1):
                if i == k:
                    A[j][k] = fraction(0,1)
                    L[j][k] = fraction(-1,1)*temp
                else:
                    A[j][k] = A[j][k] + temp*A[i][k]

    U = A[:] #upper matrix
    
    return [L,U,P]

def float_to_fraction(num): #para utilizar float, junto a fracoes, defini a funcao para transformar float em fracao
    num = num.as_integer_ratio()
    return fraction(num[0],num[1])

def matrix_float_to_fraction(x):
    for i in range (len(x)):
        for j in range (len(x[0])):
            if isinstance(x[i][j], float):
                x[i][j] = float_to_fraction(x[i][j])
            else:                  
                x[i][j] = fraction(x[i][j],1)
    return x

def print_matriz(A):
    n = len(A)
    for i in range(n):
        print(A[i])

def resolve_sistema(L, U, vetor_b):
    sistema = []
    
    if L == 0:
        sistema = U[:]
        sistema = MontaSistema(sistema, vetor_b)
    elif U == 0:
        sistema = L[:]
        sistema = flip(sistema)
        vetor_b = flip(vetor_b)
        sistema = MontaSistema(sistema, vetor_b)
    else:
        return

    resultado = []
    for i in range(len(sistema)):
        resultado.insert(0, 0)
    for i in range(len(sistema)-1,-1,-1):
        resultado[i] = sistema[i][len(sistema)]/sistema[i][i]
        for j in range(i-1,-1,-1):
            sistema[j][len(sistema)] = sistema[j][len(sistema)] - sistema[j][i]*resultado[i]
    for i in range(len(resultado)):
        resultado[i] = [resultado[i]]

    if U == 0:
        resultado = flip(resultado)
        
    return resultado
            
def main():
    
    x = [[1,2,3,4],[3,3,5,7],[2,4,6,6],[4,3,21,1]]
    x = matrix_float_to_fraction(x)

    print("A =")
    print_matriz(x)
    print()
    print("--------------------------------------------------------------------------")
    print()
    

   
    b = luDecompose(x)
    L = b[0]
    U = b[1]
    P = b[2]
    while True:
        print("L =")
        print_matriz(L)
        print()
        print("U =")
        print_matriz(U)
        print()
        print("P =")
        print_matriz(P)
        print()
        print("L.U =")
        print_matriz(MultiplicaMatriz(L,U))
        print()
        print("P.(L.U) =")
        print_matriz(MultiplicaMatriz(P,MultiplicaMatriz(L,U)))
        print()
                  

        continua = str(input("Deseja resolver o sistema? (s)/(n): "))
        if continua == ("n" or "N" or "nao" or "Nao" or "NAO"):
            print("**FIM DO PROGRAMA**")
            return 
        elif continua == ("s" or "S" or "sim" or "Sim" or "SIM"):
            print()
        else:
            return exit("ERROR")


        n = len(x)
        
        vetor_b = []
        for i in range(n):
            vetor_b.append([])
            
        trig = -1
        for i in range(n):
            y = float(input("Diga o vetor resultado b, elemento por elemento: "))
            vetor_b[i].append(y)
        print()
        vetor_b = matrix_float_to_fraction(vetor_b)
        print("Ax =", vetor_b)
        print()
        print("Ax = Pt.b <=> L.Ux = Pt.b")
        print("LUx = Pt.b <=> Ly = Pt.b")
        vetor_b = MultiplicaMatriz(MatrizTransp(P),vetor_b)
        vetor_c = resolve_sistema(L, 0, vetor_b)
        print()
        print("Ux =", vetor_c)
        print()
        resultado = resolve_sistema(0,U,vetor_c)
        print("Vetor resultado =", resultado)
        print()
        print("--------------------------------------------------------------------------")
        print()
        continua = str(input("Deseja continuar o programa? (s)/(n): "))
        if continua == ("n" or "N" or "nao" or "Nao" or "NAO"):
            print("**FIM DO PROGRAMA**")
            return 
        elif continua == ("s" or "S" or "sim" or "Sim" or "SIM"):
            print()
        else:
            return exit("ERROR")
              
        

    

main()

  


