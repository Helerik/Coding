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



################################################################################################################################
################################################################################################################################

## Objetivos: funcoes de resolucao e analise de matrizes por metodo de gauss, minimos quadrados e decomposicao P*L*U = A .
## Foi criada anteriormente a classe Fraction, para treino e tambem melhorar os arredondamentos de resultado de operacoes, provenientes
## de aproximacoes do Python.



from sys import exit #necessario para erros

## Cria a transposta da matriz dada a partir de uma matriz inicial:
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

## Multiplica duas matrizes, na ordem dada:
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

    ## Cria Matriz Sistema, juntando os dois parametros iniciais (nao fez muito sentido, pode ser implementado de forma diferente)
    ## OBS: pode ser util, se tudo o que eu quero eh resolver a matriz, sem um resultado fixo (metodo P.L.U = A).
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



## Resolve sistema em forma de matriz pelo metodo da eliminacao de Gauss
def Gauss( sistema, igualdades):                
    
    sistema = MontaSistema(sistema,  igualdades)

    ## Resolve Matriz Sistema
    
    P = [] #P eh a matriz permutacao. Ela precisa ser pre-gerada antes das operacoes de triangularizacao,
           #pois sera montada ao longo das seguintes etapas (inicialmente eh a matriz identidade)
    for i in range(len(sistema)):
        P.insert(i, [])
        for j in range(len(sistema[i])-1):
            P[i].append(fraction(0,1))

    for i in range(len(sistema)):
        for j in range(len(sistema[i])-1):
            if i == j:
                P[i][j] = fraction(1,1)
            else:
                P[i][j] = fraction(0,1)
                
    L = [] #lower matrix
    for i in range(len(sistema)):
        L.insert(i, [])
        for j in range(len(sistema[0])-1):
            if j == i:
                L[i].append(fraction(1,1))
            else:
                L[i].append(fraction(0,1))
           
    for i in range(len(sistema)):
        Maxcol = abs(sistema[i][i]) #encontro o pivo
        Maxlin = i                  #defino a linha que possui maior valor na coluna dada
        for j in range(i+1, len(sistema)):
            if abs(sistema[j][i]) > Maxcol:
                Maxcol = abs(sistema[j][i])
                Maxlin = j
            
        #troca maior linha com linha atual, coluna por coluna
        for j in range(i, len(sistema)+1):
            
            tmp = sistema[Maxlin][j] 
            sistema[Maxlin][j] = sistema[i][j] 
            sistema[i][j] = tmp


        tmp = P[i] #crio a matriz P, permutando a matriz identidade
        P[i] = P[Maxlin]
        P[Maxlin] = tmp

        

       
        #zera coluna
        for j in range(i+1, len(sistema)):
            
            temp = fraction(-1,1)*sistema[j][i]/sistema[i][i] #utiliza o pivo para zerar a coluna abaixo (no processo, altera os valores do sistema
                                                              #e, dependendo, do resultado, para encontrar o vetor.
            for k in range(i, len(sistema)+1, 1):
                if i == k:
                    sistema[j][k] = temp
                    L[j][k] = temp
                else:
                    sistema[j][k] = sistema[j][k] + temp*sistema[i][k]

    print(sistema)
    U = [] #upper matrix

    for i in range(len(sistema)):
        U.insert(i, [])
        for j in range(len(sistema[0])-1):
            U[i].append(fraction(0,1))
        


    #duas funcoezinhas para decompor a matriz A (sistema)
    count = -1
    for i in range(len(U)):
        count +=1
        for j in range(count, len(U[0]), 1):
            U[i][j] = sistema[i][j]

    resultado = []
    for i in range(len(sistema)):
        resultado.insert(0, 0)

    for i in range(len(sistema)-1,-1,-1):
        resultado[i] = sistema[i][len(sistema)]/sistema[i][i]
        for j in range(i-1,-1,-1):
            sistema[j][len(sistema)] = sistema[j][len(sistema)] - sistema[j][i]*resultado[i]

    
    
    return [resultado,U,L,P] #a funcao retorna o vetor com o resultado para o sistema (vetor x), matriz U, matriz L e matriz P.

               
def MMQ(sistema, Igualdades): #metodo alternativo para resolucao de sistemas; encontra solucoes de melhor aproximacao de sistemas sem solucao

    M = sistema
    Ig = Igualdades
    
    Mt = MatrizTransp(M)
    MtXM = MultiplicaMatriz(Mt,M)
    MtXIg = MultiplicaMatriz(Mt,Ig)

    return Gauss(MtXM, MtXIg)


def float_to_fraction(num): #para utilizar float, junto a fracoes, defini a funcao para transformar float em fracao
    num = num.as_integer_ratio()
    return fraction(num[0],num[1])

def print_resultado(resultado, U, L, P, A): #apenas um front-end para exibir melhor o resultado e apresentar todos os dados encontrados
    
    print()
    print("Matriz U =",U)
    print()
    print("Matriz L =",L)
    print()
    print("Matriz P =",P)
    print()
    print("--------------------------------------------------------------------------")
    print()
    print("Vetor resultado:", resultado)
    print()
    for i in range(len(resultado)):
        c = str(resultado[i]).split('/')
        d = int(c[0])/int(c[1])
        print("x_%i"%i,"=",d)
    print()
    print("L.U =", MultiplicaMatriz(L,U), "=", MultiplicaMatriz(P,A))
    print()
    print("--------------------------------------------------------------------------")


def main():
    
    x = [[1,2,3,4],[3,3,5,7],[2,4,6,6],[4,3,21,1]]
    y = [[1],[1],[1],[1]]

    print("A =", x,"=", y)
    print("--------------------------------------------------------------------------")


    
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
                
#############################################################    
    b = Gauss(x,y)
    print_resultado(b[0],b[1],b[2],b[3], x)
#############################################################    



            
        

main()

##### IMPORTANTE!!! #####
##### CORRIGIR ERRO NA GERACAO DA MATRIZ P. ELA RODA SOMENTE DESCENDO A DIAGONAL, LOGO, NAO TROCA TODAS AS LINHAS QUE DEVEM SER TROCADAS!!! #####


##Obs finais: preciso criar outras formas e funcoes para encontrar os resultados desejados.
##Ha o que implementar ainda, enquanto o curso avanca. Posso adicionar um metodo para encontrar matriz inversa, baseado em LU









