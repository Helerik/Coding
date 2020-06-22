from math import cos, sin, e, pi, log, sqrt, factorial
import random as r
import time
import scipy as sp
from scipy.stats import beta
import matplotlib.pylab as plt
import numpy as np

################################
################################

# funcao que sera integrada
def f(x):
    a = 0.50886257 #RG
    b = 0.10736584 #NUSP
    return (np.exp(-a*x))*np.cos(b*x) #e^(-ax)*cos(bx)

def integral(f,a,b,n): #integral de riemman no intervalo [a,b] com n intervalos
    count = 0
    integral = 0
    interval = b-a
    delta = interval/n
    while count < n:
        integral += (f(a+count*delta)+f(a+(count+1)*delta))*(delta/2)
        count += 1
    return integral

def gama(f): #gama real, aproximado por soma de riemann 
    return integral(f,0,1,100000)   # nao eh realmente utilizado, mas esta aqui por questao de testes


## crud Monte Carlo method:
def gamac(f,n): #somatoria de f(xi), xi~U[0,1], dividido por n. Aproxima-se para n grande da integral (eh como uma media)
    soma = 0
    count = 0
    while count <= n:
        soma += f(r.uniform(0,1))
        count += 1
    soma = soma/n
    return soma

def S2(gama, n): #variância amostral de um dado gama, para n amostras. Toma um gama pre-calculado.
    s2 = 0
    count = n
    while count >= 0:
        x = r.uniform(0,1)
        s2 += (x - gama)**2
        count -= 1
    s2 = s2/n
    return s2

def find_n_crud(f,erro): #metodo para encontrar um n suficientemente grande para que o erro seja menor do que 0.01 (1,00%)
    t = time.time()
    z = 2.575 #confiança 99% para distribuição normal -> gama eh uma media, e as medias tem distribuicao normal
    n = 1
    G = gamac(f,n)
    s2 = S2(G, n)
    S = sqrt(s2) #desvio padrao estimado
    e = z*S/(sqrt(n)) #erro estimado e (estimado, pois a variancia tambem eh estimada
    while e >= erro: #erro eh recalculado para n's diferentes, ate encontrar um erro desejado
        n *= int(sqrt(10))*2 #para diminuir o erro em 10 vezes, eh necessario aumentar o n 100 vezes, pois e = z*S/sqrt(n)
                             #pelo bem do tempo de computacao e de manter o n de tamanho razoavel, aumento n por um fator menor do que 100 vezes.
        G = gamac(f,n)
        s2 = S2(G, n)
        S = sqrt(s2)
        e = z*S/(sqrt(n))
        # assumo que há um erro para o meu erro, logo faço uma correção (O erro eh estimado. O erro real pode ser maior do que minha estimativa...
    if e >= 9*erro/10:
        n = int(n+n/(s2/erro))
        G = gamac(f,n)
        s2 = S2(G, n)
        S = sqrt(s2)
        e = z*S/(sqrt(n))
        
    print("Tempo de computação: ", time.time() - t)
    return [e, s2, n, G] #retorna valores relevantes encontrados

def find_n_crud_2(f,erro): ### METODO ALTERNATIVO ###
                           # Utiliza um gama com 100 vezes mais amostras do que o gama estimado, e imita gama real com esse gama n*100
                           # Por desconhecer qualquer justificativa do porque esse metodo funciona tao bem, nao o utilizei de fato,
                           # mas ele da resultados bem precisos. * erro = |G-G2|/G2 *
    t = time.time()
    n = 1
    n2 = 100
    G = gamac(f,n)
    G2 = gamac(f,n2)
    e = abs(G2 - G)/G2
    while e >= erro:
        n *= int(sqrt(10))*2
        n2 = 100*n
        G = gamac(f,n)
        G2 = gamac(f,n2)
        e = abs(G2 - G)/G2
        # assumo que há um erro para o meu erro, logo faço uma correção
    if e >= 9*erro/10:
        n *= int(sqrt(10))
        n2 = 10*n
        G = gamac(f,n)
        G2 = gamac(f,n2)
        e = abs(G2 - G)/G2
        
    print("Tempo de computação: ", time.time() - t)
    
    return [e, n+n2 , G] #como faco a amostragem de dois gamas, o N total de amostras eh n+n2


## Hit-or-Miss method:
def gamah(f,n): #faz a proporcao de todos os pontos (x,y) uniformemente distribuidos que caem entre a curva e o eixo x.
                #Para n grande, aproxima a area da curva.
    i = 0
    counter = 0

    while i <= n:
        x = r.uniform(0,1)
        y = r.uniform(0,1)
        if y<=f(x):
            counter += 1
        i += 1
    return counter/n

def find_n_hit(f,erro):
    # hit or miss gera um gama que tem cara de proporção (ou porcentagem),
    # por isso, considero como melhor alternativa para estimá-lo, um sigma
    # para p, calculado como (p(p-1))/n, onde p = gama estimado.
    # por se tratar de uma proporção, o n deve ser grande para mostrar um
    # resultado minimamente correto e não fazer conclusões erradas.
    # como quero um erro de pelo menos 1% (0.01), a amostra deve ter ao menos
    # tamanho 100 aproximadamente.

    # Semelhante ao metodo anterior, porem, a variancia possui um pouco menos de incerteza, uma vez que nao eh calculado baseado em xi's aleatorios,
    # alem dos de gama.
    t = time.time()
    z = 2.575
    n = 1/erro
    G = gamah(f,n)
    e = z*sqrt(abs(G*(1-G))/n)
    while e >= erro-0.1*erro:
        n *= int(sqrt(10))
        G = gamah(f,n)
        e = z*sqrt(abs(G*(1-G))/n)
        
    print("Tempo de computação: ", time.time() - t)
    
    return [e, sqrt((G*(1-G))/n), n, G]

## Importance Sampling method:

a, b = 0.65 , 1 #parametros para beta

def g(x,a,b): #funcao util para o metodo
    return beta.pdf(x, a, b)

def gamas(f,g,n): #faz a somatoria de f(xi)/g(xi), com xi~g(x), dividido pir n. Serve para reduzir a variancia, consequentemente, o erro.
    # g(x) = beta.pdf(x)
    i = 0
    soma = 0
    while i < n:
        x = np.random.beta(a,b)
        soma += f(x)/g(x,a,b)
        i += 1
    return soma/n

def S2s(gama, n, a, b): #a variancia se baseia em uma amostra aleatoria com distribuicao beta, ao inves de uniforme
    s2 = 0
    count = n
    while count >= 0:
        x = np.random.beta(a,b)
        s2 += ((f(x))/(g(x,a,b)) - gama)**2
        count -= 1
    s2 = s2/n
    return s2

def find_n_smp(f,g,erro):
    t = time.time()
    z = 1.96 #confiança 95% para distribuição normal (notei que se fosse muito grande, o n acabava sendo muito pequeno, gerando, ironicamente, erros maiores)
    n = 1
    G = gamas(f,g,n)
    s2s = S2s(G, n, a, b)
    S = sqrt(s2s)
    e = z*S/(sqrt(n))
    while e >= erro:
        n *= int(sqrt(10))*2
        G = gamas(f,g,n)
        s2s = S2s(G, n, a, b)
        S = sqrt(s2s)
        e = z*S/(sqrt(n))
        # assumo que há um erro para o meu erro, logo faço uma correção
    if e >= 9*erro/10:
        n = int(n+n/(s2s/erro))
        G = gamas(f,g,n)
        s2s = S2s(G, n, a, b)
        S = sqrt(s2s)
        e = z*S/(sqrt(n))
        
    print("Tempo de computação: ", time.time() - t)
    return [e, s2s, n, G]
    

def plot(f,g,a,b): #funcao auxiliar para geracao de graficos e pontos aleatorios
    t = np.arange(0.,3.,0.01)
    
    x = np.random.beta(a,b, size=100)
    y = x/x
    fig, ax = plt.subplots()
    ax.scatter(x,y)

    axes = plt.gca()
    axes.set_xlim([0.,3.])
    axes.set_ylim([0.,3.])

    plt.plot(t, f(t),t,g(t,a,b))
    plt.show()


## Control Variance method:
def fi(x): # e^u = 1 + u + u^2/2! + u^3/3! + ...
           # e^(-ax) = 1 + (-ax) + (-ax)^2/2! + ...
    i = 0
    soma = 0
    while i < 6:
        soma += ((-0.50886257*x)**i)/factorial(i)
        i += 1
    return soma*cos(0.10736584*x)

def gama_f(f,n):
    return gamac(f,n)

def gama_fi(fi, n):
    return gamac(fi,n)

def int_fi(fi):
    return integral(fi, 0, 1, 100000)

def Variancia(gama, n):
    return S2(gama, n)

def Covar(gama_f, gama_fi, n): #covariancia amostral
    i = 0
    soma = 0
    Gf = gama_f
    Gfi = gama_fi
    while i < n:
        x = r.uniform(0,1)
        x1 = r.uniform(0,1)
        soma += (x - Gf)*(x1 - Gfi)
        i += 1
    return soma/n

def gamavar(gama_f, gama_fi, fi, n):
    return gama_f - ((Covar(gama_f, gama_fi, n))/Variancia(gama_f,n))*(gama_fi - int_fi(fi)) #gama do metodo

def VarSub(gama_f, gama_fi, n):
    return Variancia(gama_f,n) - ((Covar(gama_f, gama_fi, n))**2)/Variancia(gama_fi,n) #variancia reduzida

def find_n_var(f,fi,erro): #igual aos outros metodos
    t = time.time()
    z = 2.575 #confiança 99% para distribuição normal
    n = 1
    Gf = gama_f(f, n)
    Gfi = gama_fi(fi, n)
    S2 = abs(VarSub(Gf, Gfi, n)) #uso valor absoluto, pois nesse metodo, pode-se obter por erros da amostra, valores negativos
    S = sqrt(S2)
    e = z*S/sqrt(n)
    while e >= erro:
        n *= int(sqrt(10))*2
        Gf = gama_f(f, n)
        Gfi = gama_fi(fi, n)
        S2 = abs(VarSub(Gf, Gfi, n))
        S = sqrt(S2)
        e = z*S/sqrt(n)
    if e >= 9*erro/10:
        n = int(n+n/(s2s/erro))
        Gf = gama_f(f, n)
        Gfi = gama_fi(fi, n)
        S2 = abs(VarSub(Gf, Gfi, n))
        S = sqrt(S2)
        e = z*S/(sqrt(n))

    GF = gamavar(Gf, Gfi, fi, n)
        
    print("Tempo de computação: ", time.time() - t)
    return [e, S2, n, GF]




def main(): #front-end para display dos valores
    while True:
        print("** Inicio **")
        print("------------")
        print()
        print("Metodos de Monte Carlo para estimacao da integral de f(x) = cos(0.10736584x) . e^(0.50886257x)")
        print()
        print("** Metodo Crud **")
        print()
        GAM = find_n_crud(f, 0.01)
        ec = GAM[0] 
        vc = GAM[1] 
        nc = GAM[2] 
        gamc = GAM[3] 
        print()
        print("Erro estimado (desejado <= 1%): ",ec*100)
        print("Variancia amostral: ", vc)
        print("N = ", nc)
        print("Gama (integral) estimado: ", gamc)
        print()
        print("--------------------")
        print()
        print("** Metodo Hit-or-Miss **")
        print()
        GAM = GAM = find_n_hit(f, 0.01)
        eh = GAM[0]  
        vh = GAM[1] 
        nh = GAM[2] 
        gamh = GAM[3] 
        print()
        print("Erro estimado (desejado <= 1%): ",eh*100)
        print("Variancia amostral de proporcoes: ", vh)
        print("N = ", nh)
        print("Gama (integral) estimado: ",gamh)
        print()
        print("--------------------")
        print()
        print("** Importance Sampling **")
        print()
        GAM = GAM = find_n_smp(f, g, 0.01)
        es = GAM[0] 
        vs = GAM[1]
        ns = GAM[2]
        gams = GAM[3] 
        print()
        print("g(x) = beta(x, 0.65, 1)")
        print()
        print("Erro estimado (desejado <= 1%): ",es*100)
        print("Variancia amostral: ", vs)
        print("N = ", ns)
        print("Gama (integral) estimado: ", GAM[3])
        print()
        print("** Crud x Importance Sampling **")
        print()
        print("N = ", nc)
        print()
        G = gamas(f, g, nc)
        print("Gama crud = ", gamc, "x Gama IS = ", G)
        print("Var. crud = ", vc, "x Var. IS = ", S2s(G, nc, a, b))
        print()
        print("--------------------")
        print()
        print("** Control Variate **")
        print()
        GAM = GAM = find_n_smp(f, g, 0.01)
        ev = GAM[0] 
        vv = GAM[1]
        nv = GAM[2]
        gamv = GAM[3] 
        print()
        print("Erro estimado (desejado <= 1%): ",ev*100)
        print("Variancia amostral: ", vv)
        print("N = ", nv)
        print("Gama (integral) estimado: ", gamv)
        print()
        print("--------------------")
        print()
        

        end = str(input("** Deseja recalcular? [s]/[n] ** \n"))
        if end == "n":
            print()
            print("** Fim do programa **")
            time.sleep(1.0)
            print()
            return 
                  





main()
