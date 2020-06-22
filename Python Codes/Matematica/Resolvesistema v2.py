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

    def getnumerador(self):
        return self._numerador

    def getdenominador(self):
        return self._denominador
    
    def __init__(self, numerador, denominador):
        
        if (not isinstance(numerador, int)):
            raise (TypeError("O numerador deve ser inteiro, ou uma fracao"))
        if (not isinstance(denominador, int)):
            raise (TypeError("O denominador deve ser inteiro, ou uma fracao"))

        if isinstance(numerador, fraction) or isinstance(denominador, fraction):
            return numerador

        if denominador == 0:
            raise ZeroDivisionError("O denominador da fracao nao pode ser zero")

        if numerador == 0:
            self._numerador = 0
            self._denominador = 1
        else:
            if (numerador < 0 and denominador >= 0) or (numerador >=0 and denominador <0):
                sign = -1
            else:
                sign = 1
            (self._numerador, self._denominador) = self._reduce(numerador, denominador, sign)

    def __repr__(self):
        return str(self._numerador)+"/"+str(self._denominador)

    def __eq__(self, direita):
        esquerda = self
        if esquerda.getnumerador() == direita.getnumerador() and esquerda.getdenominador() == direita.getdenominador():
            return True
        else:
            return False

    def __ne__(self, direita):
        esquerda = self
        return not esquerda == direita

    def __lt__(self, direita):
        esquerda = self
        return (esquerda.getnumerador() * direita.getdenominador()) < (esquerda.getdenominador() * direita.getnumerador())

    def __le__(self, direita):
        esquerda = self
        return not direita < esquerda

    def __gt__(self, direita):
        esquerda = self
        return direita < esquerda

    def __ge__(self, direita):
        esquerda = self
        return not direita > esquerda

    def __add__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() * direita.getdenominador() + direita.getnumerador() * esquerda.getdenominador()
        den = direita.getdenominador() * esquerda.getdenominador()
        return fraction(num,den)

    def __sub__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() * direita.getdenominador() - direita.getnumerador() * esquerda.getdenominador()
        den = direita.getdenominador() * esquerda.getdenominador()
        return fraction(num,den)

    def __mul__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() * direita.getnumerador()
        den = esquerda.getdenominador() * direita.getdenominador()
        return fraction(num,den)

    def __truediv__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() *  direita.getdenominador()
        den = esquerda.getdenominador() * direita.getnumerador()
        return fraction(num,den)

    def __abs__(self):
        num = abs(self.getnumerador())
        den = abs(self.getdenominador())
        return fraction(num, den)



## Resolucao de Sistemas por metodo dos minimos quadrados ##

from sys import exit

## Cria a transposta da matriz dada:

def MatrizTransp (matriz):

    M = matriz
    
    lin = len(M) # refere a j
    col = len(M[0]) # refere a i

    transp = []

    for i in range(col):
        transp.insert(i, [])


    for j in range(lin):
        for i in range(col):
            transp[i].insert(j, M[j][i])

    return transp

## Multiplica duas matrizes

def MultiplicaMatriz(matriz1, matriz2):

    M1 = matriz1
    M2 = matriz2
    
    lin1 = len(M1)
    col1 = len(M1[0])
    lin2 = len(M2)
    col2 = len(M2[0])

    MXM = [] ## matriz multiplicada
    for i in range(lin1):
            MXM.insert(i, [])
            for j in range(col2):
                MXM[i].insert(j, fraction(0,1))

    if col1 != lin2:
        return exit("ERROR: Tried to multiply incompatible matrices; Verify matrices multiplication order or scale (i.e. mxn)")

    for i in range (lin1):
        for j in range (col2):
            for k in range (lin2):
                MXM[i][j] = (MXM[i][j]) + ((M1[i][k])*(M2[k][j]))
   
    return MXM

    ## Cria Matriz Sistema, juntando os dois parametros iniciais (nao fez muito sentido, pode ser implementado de forma diferente)
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



## Resolve sistema em forma de matriz pelo metodo de Gauss

def Gauss( sistema, igualdades):

    sistema = MontaSistema(sistema,  igualdades)

    ## Resolve Matriz Sistema
                
    for i in range(len(sistema)):

        # max coluna
        Maxcol = abs(sistema[i][i])
        Maxlin = i
        for j in range(i+1, len(sistema)):
            if abs(sistema[j][i]) > Maxcol:
                Maxcol = abs(sistema[j][i])
                Maxlin = j
                
        #troca maior linha com linha atual
        for j in range(i, len(sistema)+1):
            tmp = sistema[Maxlin][j]
            sistema[Maxlin][j] = sistema[i][j]
            sistema[i][j] = tmp

        #zera coluna
        for j in range(i+1, len(sistema)):
            temp = fraction(-1,1)*sistema[j][i]/sistema[i][i]
            for k in range(i, len(sistema)+1):
                if i == k:
                    sistema[j][k] = 0
                else:
                    sistema[j][k] = sistema[j][k] + temp*sistema[i][k]

    
    resultado = []
    for i in range(len(sistema)):
        resultado.insert(0, 0)

    for i in range(len(sistema)-1,-1,-1):
        resultado[i] = sistema[i][len(sistema)]/sistema[i][i]
        for j in range(i-1,-1,-1):
            sistema[j][len(sistema)] = sistema[j][len(sistema)] - sistema[j][i]*resultado[i]


    return resultado

               
def MMQ(sistema, Igualdades):

    M = sistema
    Ig = Igualdades
    
    Mt = MatrizTransp(M)
    MtXM = MultiplicaMatriz(Mt,M)
    MtXIg = MultiplicaMatriz(Mt,Ig)

    return Gauss(MtXM, MtXIg)

def float_to_fraction(num):
    
    if isinstance(num, int):
        return fraction(num, 1)
    
    else:
        count = 1
        while num.is_integer() == False:
            num = num*10
            count = count*10
        num = int(num)

        return fraction(num, count)

def main():
    x = [[3,-1,-2],[2,2,5],[1,3,7]]
    y = [[3],[-1],[2]]
    
    for i in range (len(x)):
        for j in range (len(x[0])):
            if isinstance(x[i][j], float):
                x[i][j] = float_to_fraction(x[i][j])
            else:                  
                x[i][j] = fraction(x[i][j],1)
    for i in range (len(y)):
        for j in range (len(y[0])):
            if isinstance(y[i][j], float):
                y[i][j] = float_to_fraction(y[i][j])
            else:                  
                y[i][j] = fraction(y[i][j],1)
    
    b = MMQ(x,y)
    print(b)
    for i in range(len(b)):
        c = str(b[i]).split('/')
        d = int(c[0])/int(c[1])
        print(d)
    
            
        

main()











