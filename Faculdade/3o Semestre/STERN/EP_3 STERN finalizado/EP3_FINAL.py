from math import cos, sin, e, pi, log, sqrt, factorial
import random as r
import time
import scipy as sp
from scipy.stats import beta
import matplotlib.pylab as plt
import numpy as np
##import sobol_seq as sobol
from VanDerCoput import vdc

##IMPLEMENTACAO IGUAL A DO EP 2, POREM COM GERADORES QUASI-RANDOM AO INVES DE PSEUDO-RANDOM:


################################################################################################
################################################################################################

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

def sobol_quasi_r(n):
    if n>40:
        dif = n-40
        sob = sobol.i4_sobol(40, int(10000*r.random()))
        sob = sob[0]
        sob2 = sobol_quasi_r(dif)
        sob = np.append(sob, sob2)
    else:
        sob = sobol.i4_sobol(n,int(10000*r.random()))
        sob = sob[0]
    return sob

def Vander(n):
    van = [vdc(i) for i in range(n)]
    return van

###########################################################################################################

## crud Monte Carlo method:

# SEQUENCIAS DE SOBOL:
def gamac_sob(f,x,n): #somatoria de f(xi), onde xi eh um valor de uma sequencia Quasi-Random de Sobol.
    soma = 0
    count = 0
    while count < n:
        soma += f(x[count])
        count += 1
    soma = soma/n
    return soma

def S2_sob(gama,x, n): #variância amostral de um dado gama, para n amostras. Toma um gama pre-calculado.
    s2 = 0
    count = n-1
    while count >= 0:
        s2 += (x[count] - gama)**2
        count -= 1
    s2 = s2/n
    return s2

def find_n_crud_sob(f,erro): #metodo para encontrar um n suficientemente grande para que o erro seja menor do que 0.01 (1,00%)
    t = time.time()
    z = 2.575 #confiança 99% para distribuição normal -> gama eh uma media, e as medias tem distribuicao normal
    n = 1
    x = sobol_quasi_r(n)
    G = gamac_sob(f,x,n)
    s2 = S2_sob(G,x,n)
    S = sqrt(s2) #desvio padrao estimado
    e = z*S/(sqrt(n)) #erro estimado e (estimado, pois a variancia tambem eh estimada
    while e >= erro: #erro eh recalculado para n's diferentes, ate encontrar um erro desejado
        n *= 10
        x = sobol_quasi_r(n)
        G = gamac_sob(f,x,n)
        s2 = S2_sob(G,x,n)
        S = sqrt(s2)
        e = z*S/(sqrt(n))
        if round(e, 3) == erro:
            break
        
    print("Tempo de computação: ", time.time() - t)
    return [e, s2, n, G] #retorna valores relevantes encontrados

##########################################################################################################

#VanDerCorput
def gamac_van(f,x,n): #somatoria de f(xi), onde xi eh um dos valores de uma sequencia Quasi-Random VanDerCorput
    soma = 0
    count = 0
    while count < n:
        soma += f(x[count])
        count += 1
    soma = soma/n
    return soma

def S2_van(gama,x, n): #variância amostral de um dado gama, para n amostras. Toma um gama pre-calculado.
    s2 = 0
    count = n-1
    while count >= 0:
        s2 += (x[count] - gama)**2
        count -= 1
    s2 = s2/n
    return s2

def find_n_crud_van(f,erro): #metodo para encontrar um n suficientemente grande para que o erro seja menor do que 0.01 (1,00%)
    t = time.time()
    z = 2.575 #confiança 99% para distribuição normal -> gama eh uma media, e as medias tem distribuicao normal
    n = 1
    x = Vander(n)
    G = gamac_van(f,x,n)
    s2 = S2_van(G,x,n)
    S = sqrt(s2) #desvio padrao estimado
    e = z*S/(sqrt(n)) #erro estimado e (estimado, pois a variancia tambem eh estimada
    while e >= erro: #erro eh recalculado para n's diferentes, ate encontrar um erro desejado
        n *= int(sqrt(10))*2
        x = Vander(n)
        G = gamac_van(f,x,n)
        s2 = S2_van(G,x,n)
        S = sqrt(s2)
        e = z*S/(sqrt(n))
        
    print("Tempo de computação: ", time.time() - t)
    return [e, s2, n, G] #retorna valores relevantes encontrados

####################################################################################################################################

## Hit-or-Miss method:

#VanDerCorput
def gamah_van(f,x,y,n): #faz a proporcao de todos os pontos (x,y) uniformemente distribuidos que caem entre a curva e o eixo x.
                #Para n grande, aproxima a area da curva.
    i = 0
    counter = 0
    while i < n:
        if y[i]<=f(x[i]):
            counter += 1
        i += 1
    return counter/n

def find_n_hit_van(f,erro):
    # hit or miss gera um gama que tem cara de proporção (ou porcentagem),
    # por isso, considero como melhor alternativa para estimá-lo, um sigma
    # para p, calculado como (p(p-1))/n, onde p = gama estimado.
    # por se tratar de uma proporção, o n deve ser grande para mostrar um
    # resultado minimamente correto e não fazer conclusões erradas.
    # como quero um erro de pelo menos 1% (0.01), a amostra deve ter ao menos
    # tamanho 100 aproximadamente.
    t = time.time()
    z = 2.575
    n = int(1/erro)
    x = Vander(n)
    y = [i/n for i in range(n)]
    G = gamah_van(f,x,y,n)
    e = z*sqrt(abs(G*(1-G))/n)
    while e >= erro-0.1*erro:
        n *= int(sqrt(10))
        x = Vander(n)
        y = [i/n for i in range(n)]
        G = gamah_van(f,x,y,n)
        e = z*sqrt(abs(G*(1-G))/n)
        
    print("Tempo de computação: ", time.time() - t)
    return [e, sqrt((G*(1-G))/n), n, G]

##################################################################################################################################

## Importance Sampling method:

def gerador_(n):
    x = Vander(n)
    res = [0 for i in range(len(x))]
    counter = 0
    for i in range(len(x)):
        counter -= 1
        res[counter] = G_inv(x[i])
    return res

def g(x): #funcao util para o metodo
    return -0.40228*x+1.20114

def G_inv(x):
    return 2.98583 - 0.0000497166*sqrt(3.60684*(10**9) - (2.0114*(10**9))*x)

def gamas_van(f,g,x,n): #faz a somatoria de f(xi)/g(xi), com xi~g(x), dividido por n. Serve para reduzir a variancia, consequentemente, o erro.
    i = 0
    soma = 0
    while i < n: 
        soma += f(x[i])/g(x[i])
        i += 1
    return soma/n

def S2s(gama, x, n): #a variancia se baseia em uma amostra aleatoria com distribuicao beta, ao inves de uniforme
    s2 = 0
    count = n-1
    while count >= 0:
        s2 += ((f(x[count]))/(g(x[count])) - gama)**2
        count -= 1
    s2 = s2/n
    return s2

def find_n_smp_van(f,g,erro):
    t = time.time()
    z = 1.96 #confiança 95% para distribuição normal (notei que se fosse muito grande, o n acabava sendo muito pequeno, gerando, ironicamente, erros maiores)
    n = 2
    x = gerador_(n)
    G = gamas_van(f,g,x,n)
    s2s = S2s(G,x, n)
    S = sqrt(s2s)
    e = z*S/(sqrt(n))
    while e >= erro:
        n *= int(sqrt(10))*2
        x = gerador_(n)
        G = gamas_van(f,g,x,n)
        s2s = S2s(G,x, n)
        S = sqrt(s2s)
        e = z*S/(sqrt(n))
     
    print("Tempo de computação: ", time.time() - t)
    return [e, s2s, n, G]
    

def plot(f,g,a,b): #funcao auxiliar para geracao de graficos e pontos aleatorios
    t = np.arange(0.,3.,0.01)
    
    x = np.random.beta(a,b, size=100)

    axes = plt.gca()
    axes.set_xlim([0.,3.])
    axes.set_ylim([0.,3.])

    plt.plot(t, f(t),t,g(t,a,b))
    plt.show()

###############################################################################################################

## Control Variance method:
def fi(x): # e^u = 1 + u + u^2/2! + u^3/3! + ...
           # e^(-ax) = 1 + (-ax) + (-ax)^2/2! + ...
    i = 0
    soma = 0
    while i < 6:
        soma += ((-0.50886257*x)**i)/factorial(i)
        i += 1
    return soma*cos(0.10736584*x)

def gama_f(f,x,n):
    return gamac_van(f,x,n)

def gama_fi(fi,x, n):
    return gamac_van(fi,x,n)

def int_fi(fi):
    return integral(fi, 0, 1, 1000)

def Variancia(gama,x, n):
    return S2_van(gama,x, n)

def Covar(gama_f, gama_fi,x, n): #covariancia amostral
    i = 0
    soma = 0
    Gf = gama_f
    Gfi = gama_fi
    x1 = x
    while i < n:
        soma += (x[i] - Gf)*(x1[i] - Gfi)
        i += 1
    return soma/n

def gamavar(gama_f, gama_fi, fi,x, n):
    return gama_f - ((Covar(gama_f, gama_fi,x, n))/Variancia(gama_f,x,n))*(gama_fi - int_fi(fi)) #gama do metodo

def VarSub(gama_f, gama_fi,x, n):
    return Variancia(gama_f,x,n) - ((Covar(gama_f, gama_fi,x, n))**2)/Variancia(gama_fi,x,n) #variancia reduzida

def find_n_var_van(f,fi,erro): #igual aos outros metodos
    t = time.time()
    z = 2.575 #confiança 99% para distribuição normal
    n = 2
    x = Vander(n)
    Gf = gama_f(f,x, n)
    Gfi = gama_fi(fi,x, n)
    S2 = abs(VarSub(Gf, Gfi,x, n))
    S = sqrt(S2)
    e = z*S/sqrt(n)
    while e >= erro:
        n *= int(sqrt(10))*2
        x = Vander(n)
        Gf = gama_f(f,x, n)
        Gfi = gama_fi(fi,x, n)
        S2 = abs(VarSub(Gf, Gfi,x, n))
        S = sqrt(S2)
        e = z*S/sqrt(n)

    GF = gamavar(Gf, Gfi, fi,x, n)
        
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
        GAM = find_n_crud_van(f, 0.01)
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
        GAM = GAM = find_n_hit_van(f, 0.01)
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
        GAM = GAM = find_n_smp_van(f, g, 0.01)
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
        print("--------------------")
        print()
        print("** Control Variate **")
        print()
        GAM = GAM = find_n_var_van(f, fi, 0.01)
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


#intervalo = [1/i for i in range(1, 1000)]

##plt.plot( Vander(10),intervalo, 'go')
##plt.show()
##plt.plot(np.random.beta(a,b, size=1000),intervalo, 'rs')
##plt.show()
##plt.show()
##lista = [1,0.5,0.25,0.125,2,1,0.5]

#x = gerador_(1000)
#intervalo = [i/len(x) for i in range(len(x))]

##plt.plot(lista, intervalo, 'bo')
##plt.show()

#plt.plot(x,intervalo,'bo')
#plt.show()




