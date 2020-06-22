from math import sqrt
import time as t
import matplotlib.pyplot as plt

#lê uma matriz esparsa, dado um arquivo txt (ou outros arquivos de texto)
def read_sparse(nome):
    with open(nome, 'r') as f:
        lines = f.read().splitlines()
        while "" in lines:
            lines.remove("")
        indx = [int(num) for num in lines[0].split(' ')]
        n = indx[0]
        m = indx[1]
        mtx = [[0 for j in range(m)] for i in range(n)]
        for j in range(1, len(lines)):
            l = [num for num in lines[j].split(' ')]
            l[0] = int(l[0])
            l[1] = int(l[1])
            l[2] = float(l[2])
            mtx[l[0]-1][l[1]-1] = l[2]
            mtx[-l[0]][-l[1]] = l[2]
    
    return mtx                  

def mult_matriz(matriz1, matriz2):

    M1 = matriz1 #para facilitar, usei notacao simplificada
    M2 = matriz2
    
    lin1 = len(M1)
    col1 = len(M1[0])
    lin2 = len(M2)
    col2 = len(M2[0])
    
    MxM = [] ## matriz multiplicada (pequena funcao para cria-la, vazia)
    for i in range(lin1):
            MxM.insert(i, [])
            for j in range(col2):
                MxM[i].insert(j, 0)
    if col1 != lin2: #por definicao, eh impossivel multiplicar uma matriz a direita que tenha um numero de colunas diferente do numero de linhas
                     #de uma matriz a esquerda
        return exit("ERROR: Tried to multiply incompatible matrices; Verify matrices multiplication order or scale (i.e. mxn)")

    for i in range (lin1):
        for j in range (col2):
            for k in range (lin2):
                MxM[i][j] = (MxM[i][j]) + ((M1[i][k])*(M2[k][j])) #executa a operacao de multiplicacao de matrizes
   
    return MxM

def find_b(A):
    return mult_matriz(A, [[1] for i in range(len(A))])

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

def SOR(A, b, om, guess, tol, it):
    if not 1 <= om <= 2:
        print()
        print("!! O valor indicado para ômega não pode ser utilizado !!")
        print()
        return False
    k = 0
    n = len(b)
    x0 = []
    x = []
    x0[:] = guess
    x[:] = x0
    while k <= it:
        for i in range(n):
            soma = 0
            for j in range(n):
                if j<=(i-1):
                    soma += A[i][j]*x[j][0]
                if j >= (1+i):
                    soma += A[i][j]*x0[j][0]
            x[i][0] = (1 - om)*x0[i][0] + om*(1/A[i][i])*(b[i][0]-soma)
        r = []
        tmp = mult_matriz(A,x)
        maximum = 0
        for i in range(n):
            r.append(abs(b[i][0]- tmp[i][0]))
            if r[i] > maximum:
                maximum = r[i]
        if maximum < tol:
            return [x,k]
        
        k += 1
        for i in range(n):
            x0[i][0] = x[i][0]
    if om == 1:
        print()
        print("-- Método de Gauss-Seidel --")
    if tol == 0:
        print()
        print("Resultado para",it,"iterações:")
        print()
        return [x,k]
    print()
    print("** Número máximo de iterações atingido, para tolerância =",tol," **")
    print("Valor obtido:")
    print()
    return [x,k]

def gradconj(A, b, guess, tol, it):
    n = len(b)
    r = []
    w = []
    v = []
    x = []
    x[:] = guess
    tmp = mult_matriz(A,x)
    for i in range(n):
        r.append([b[i][0] - tmp[i][0]])
        w.append([b[i][0] - tmp[i][0]])
        v.append([b[i][0] - tmp[i][0]])

    a = 0
    for i in range(n):
        a = a + (w[i][0])**2
    k = 0
    while k <= it:
        try:
            normv = 0
            for i in range(n):
                normv = normv + (v[i][0])**2
            normv = sqrt(normv)
            if normv < tol:
                return [x, k]
            u = mult_matriz(A, v)
            soma = 0
            for i in range(n):
                soma += (v[i][0])*(u[i][0])
            t = a/soma
            for i in range(n):
                x[i][0] = x[i][0] + (v[i][0])*t
                r[i][0] = r[i][0] - (u[i][0])*t
            w[:] = r
            b = 0
            for j in range(n):
                b += (w[j][0])**2
            if abs(b) < tol:
                normr = 0
                for l in range(n):
                    normr += (r[l][0])**2
                normr = sqrt(normr)
                if normr < tol:
                    return [x,k]
            s = b/a
            for i in range(n):
                v[i][0] = w[i][0] + v[i][0]*s
            a = b
            k += 1
        except:
            print()
            print("Saída especial: [divisão por zero decorrente de baixa tolerância]")
            print()
            return [x, k]
    if k > n:
        print()
        print("Número máximo de iterações atingido")
        print()
        return [x, k] 

def print_res(mtx):
    interval = [1,1.25,1.5,1.75,2]
    tim = t.time()
    
    mat = read_sparse(mtx)
    res = gradconj(mat,find_b(mat),[[0] for i in range(len(mat))],0.000001,2000)
    it = res[1]
    res = res[0]
    dif = 0

    print("Tamanho da matriz:", len(mat))
    print()
    print("     Resultado obtido:", end='')
    print(" "*16, end = '')
    print("Resultado real:")
    for i in range(len(res)):
        print("    [%17.17f]" %res[i][0], "-"*10, "[1.0]")
        dif += (res[i][0]-1)**2
    dif = sqrt(dif)
    T = t.time() - tim
    print()
    print("    Norma2 da diferença:", dif)
    print("    Número de iterações:",it)
    print("    Tempo de computação:", T)  

def main():
    print()
    print("=" * 100)
    print()
    print("Resultados obtidos para o método Gradiente Conjugado, com um máximo de 2000 iterações e tolerância = 0.000001: ")
    print()
    print("Matriz 1: ")
    print()
    print_res("mtx1.mtx")
    print()
    print("-" * 100)
    print()
    print("=" * 100)
    print()
    print("Matriz 2: ")
    print()
    print_res("mtx2.mtx")
    print()
    print("-" * 100)
    print()
    print("=" * 100)
    print()
    print("Matriz 3: ")
    print()
    print_res("mtx3.mtx")
    print()
    print("-" * 100)
    print()
    print("=" * 100)
    print()
    print("Matriz 4: ")
    print()
    print_res("mtx4.mtx")
    print()
    print("-" * 100)
    print()
    print("=" * 100)
    print()
    print("Matriz 5: ")
    print()
    print_res("mtx5.mtx")
    print()
    print("-" * 100)
main()











