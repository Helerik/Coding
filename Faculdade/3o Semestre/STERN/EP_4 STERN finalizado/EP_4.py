# EP 4 - Markov Chain Monte Carlo

from math import cos, sin, e, pi, log, sqrt, factorial
import random as r
import time as t
import scipy as sp
from scipy.stats import gamma, norm
import matplotlib.pylab as plt
import numpy as np

################################################################################

def integral(f,a,b,n): #integral de riemman no intervalo [a,b] com n intervalos
    count = 0
    integral = 0
    interval = b-a
    delta = interval/n
    while count < n:
        integral += (f(a+count*delta)+f(a+(count+1)*delta))*(delta/2)
        count += 1
    return integral

################################################################################

# Parametros baseados respectivamente em RG e NUSP.:
C = 1.50886257
R = 1.10736584

# funcao proporcional a gamma(C,x)*abs(cos(R*x)). Isso eh; g(x) = k*gamma(C,x)*abs(cos(R*x)), onde k = 1/integral(gamma(C,x)*abs(cos(R*x))), diferente de 0.
def g(x):
    return gamma.pdf(x,C)*abs(np.cos(R*x))

# o objetivo eh integrar h(x)g(x) no intervalo de 0 a infinito. Como h(x) eh 0 fora do intervalo, eh o mesmo que
# calcular a integral no intervalo [1,2].

# kernel da cadeia de markov. Um valor aleatorio normal, com centro no x atual e sigma.
def Q(mu, sigma):
    return np.random.normal(mu, sigma)

# Algoritimo de Metropolis para MCMC:

# alpha de Metropolis-Hastings
def MCMC_M(n, S, inicio, burnin):
    #burnin
    n += burnin
    
    dist = [0 for i in range(n)]
    dist[0] = inicio
    for i in range(n-1):
        prox = Q(dist[i], S)
        if r.uniform(0,1) <= min(1, g(prox)/g(dist[i])):
            dist[i+1] = prox
        else:
            dist[i+1] = dist[i]
    for i in range(burnin):
        dist.pop(i)
    return dist

# alpha de Barker
def MCMC_B(n, S, inicio, burnin):
    #burnin
    n += burnin
    
    dist = [0 for i in range(n)]
    dist[0] = inicio
    for i in range(n-1):
        prox = Q(dist[i], S)
        if r.uniform(0,1) <= g(prox)/(g(dist[i])+g(prox)):
            dist[i+1] = prox
        else:
            dist[i+1] = dist[i]
    for i in range(burnin):
        dist.pop(i)
    return dist

# o burnin eh um metodo de corte da amostra, para que o MCMC nao seja afetado pelo valor inicial de forma muito drastica

def Z(n, S, chave, inicio, burnin):
    if chave == "M":
        x = MCMC_M(n, S, inicio, burnin)
    elif chave == "B":
        x = MCMC_B(n, S,inicio, burnin)
    soma = 0
    for i in range(len(x)):
        if 1 <= x[i] <= 2:
            soma += 1
    return soma/n

#calculo da autocorrelacao
def autocorr(data):
    n = len(data)

    media = 0
    for i in range(n):
        media += data[i]
    media = media/n

    var = 0
    for i in range(n):
        var += (data[i]-media)**2
    var = var/n

    res = []
    soma = 0
    for k in range(n//2):
        for t in range(n-k):
            soma += (data[t]-media)*(data[t+k]-media)
        soma = soma/(var*(n-k))
        res.append(abs(soma))
    
    return res

# funcao para encontrar o S, do kernel Q, baseado na autocorrelacao. Leva em conta a media das autocorrelacoes e o ponto em que cai abaixo de 0.1 e
# o quanto fica acima de 0.1 depois de passar
def find_s(m, s0, passo, burnin):
    # m = numero de iteracoes
    res = []
    S = []
    for i in range(m):
        x = MCMC_M(m, s0, 1.42, burnin)
        y = autocorr(x)
        
        ymed = 0
        mark = 0
        tempo = 0
        for j in range(len(y)):
            if y[j] <= 0.1:
                mark = j
            elif y[j] > 0.3 and mark != 0:
                tempo += 1
            ymed += y[j]
        ymed /= (len(y))
        res.append(sqrt((ymed)**2 + mark**2 + tempo**2))
        if i == 0:
            minim = sqrt(ymed**2 + mark**2 + tempo**2)
            resul = [minim, s0]
        elif sqrt(ymed**2 + mark**2 + tempo**2) < minim:
            minim = sqrt(ymed**2 + mark**2 + tempo**2)
            resul = [minim, s0]
        
        s0 += passo
        S.append(s0)
    #plt.plot(S[10:m],res[10:m])
    #plt.show()
    return resul

def s_medio(n):
    soma = 0
    for i in range(n):
        soma += find_s(100, 0.01, 0.01, int(0.1*100))[1]
    soma /= n
    return soma

# metodo alternativo para definir o S. Algumas fontes apontam que uma taxa de aceitacao de ~25% apresenta bons resultados
def MCMC_alt(n, S, inicio, burnin):
    #burnin
    n += burnin
    
    dist = [0 for i in range(n)]
    dist[0] = inicio
    acpt = 0
    for i in range(n-1):
        prox = Q(dist[i], S)
        if r.uniform(0,1) <= min(1, g(prox)/g(dist[i])):
            dist[i+1] = prox
            acpt += 1
        else:
            dist[i+1] = dist[i]
    acpt = acpt-burnin
    return acpt/n

# funcao g(x) normalizada, utilizada apenas como parametro para as minhas decisoes. O erro sera estimado, sem o uso dessa funcao
def h(x):
    return (1/0.6219348731646654)*g(x)

def find_n(S, chave, erro):
    c = chave
    n = 1000
    z1 = Z(n, S, c, 1.42, int(0.1*n))
    z2 = Z(n*10, S, c, 1.42, int(0.1*(n*10)))
    e = abs(z1-z2)/z2
    while e >= erro:
        n += 1000
        z1 = Z(n, S, c, 1.42, int(0.1*n))
        z2 = Z(n*10, S, c, 1.42, int(0.1*(n*10)))
        e = abs(z1-z2)/z2
    return [z2, e, n]

##plt.figure, plt.plot(autocorr(MCMC_M(10000, 2.3, 1.42, 1000)))
##plt.show()
##plt.figure, plt.plot(autocorr(MCMC_M(10000, 1.4, 1.42, 1000)))
##plt.show()
##
##t = np.arange(0., 10., 0.01)
##plt.figure, plt.plot(t, g(t))
##x = MCMC_M(10000, 0.5, 1.42, 1000)
##plt.figure, plt.hist(x, density=True, bins=100)
##plt.show()

res = []
soma = 0
for i in range(1,10001):
    z = Z(i, 1.4, "M", 1.42, 0)
    res.append(z)
for i in range(999,10000):
    soma += res[i]
soma /= 10000 - 999
print("Z medio para 3000 Z's =", soma)
sig2 = 0
for i in range(999, 10000):
    sig2 += (res[i] - soma)**2
sig2 /= 10000-999
print("Var =", sig2)
print("Erro com 99% de confianca =", sqrt(sig2)/sqrt(10000-999))

    

  
def main():
    print("Primeiro passo - definicao da Variancia do kernel Q:")
    print()
    print("     Metodo 1 - Avaliar a autocorrelacao:")
    print()
    S = s_medio(50)
    print("          O 'S ideal' medio encontrado, para 50 S's encontrados =", S)
    print()
    print("     Metodo 2 - Avaliar a taxa de aceitacao:")
    print()
    S2 = 2.3 #valor pre calculando usando o MCMC_alt
    print("          O S encontrado baseado na taxa de aceitacao foi =", S2)
    print()
    print("     Metodo 3 - Avaliacao da media dos metodos anteriores:")
    print()
    S3 = (S + S2)/2
    print("          O S encontrado baseado na media dos metodos anteriores foi =", S3)
    print()
    while True:
        
        print("================================================================================================")
        print()
        print("Segundo passo - calculo da integral Z com erro de no maximo 1%:")
        print()
        print("** Alpha de Metropolis-Hastings **")
        print()
        print("     Metodo 1:")
        print()
        res = find_n(S, "M", 0.01)
        z = res[0]
        eps = res[1]
        n = res[2]
        print("          Z =", z, "para n =", n)
        print("          Erro estimado =", eps)
        print("          Erro =", abs(z - integral(h, 1, 2, 10000))/integral(h, 1, 2, 10000))
        print()
        print("Plot de convergencia de Z, com n = [1:2000] (sem burn-in):")
        plt.plot([(Z(i, S, "M", 1.42, 0)) for i in range(1,2001)])
        plt.show()
        print()
        print("     Metodo 2:")
        print()
        res = find_n(S2, "M", 0.01)
        z = res[0]
        eps = res[1]
        n = res[2]
        print("          Z =", z, "para n =", n)
        print("          Erro estimado =", eps)
        print("          Erro =", abs(z - integral(h, 1, 2, 10000))/integral(h, 1, 2, 10000))
        print()
        print("Plot de convergencia de Z, com n = [1:2000] (sem burn-in):")
        plt.plot([(Z(i, S2, "M", 1.42, 0)) for i in range(1,2001)])
        plt.show()
        print()
        print("     Metodo 3:")
        print()
        res = find_n(S3, "M", 0.01)
        z = res[0]
        eps = res[1]
        n = res[2]
        print("          Z =", z, "para n =", n)
        print("          Erro estimado =", eps)
        print("          Erro =", abs(z - integral(h, 1, 2, 10000))/integral(h, 1, 2, 10000))
        print()
        print("Plot de convergencia de Z, com n = [1:2000] (sem burn-in):")
        plt.plot([(Z(i, S3, "M", 1.42, 0)) for i in range(1,2001)])
        plt.show()
        print()
        print("================================================================================================")
        print()
        print("** Alpha de Barker **")
        print()
        print("     Metodo 1:")
        print()
        res = find_n(S, "B", 0.01)
        z = res[0]
        eps = res[1]
        n = res[2]
        print("          Z =", z, "para n =", n)
        print("          Erro estimado =", eps)
        print("          Erro =", abs(z - integral(h, 1, 2, 10000))/integral(h, 1, 2, 10000))
        print()
        print("Plot de convergencia de Z, com n = [1:2000] (sem burn-in):")
        plt.plot([(Z(i, S, "B", 1.42, 0)) for i in range(1,2001)])
        plt.show()
        print()
        print("     Metodo 2:")
        print()
        res = find_n(S2, "B", 0.01)
        z = res[0]
        eps = res[1]
        n = res[2]
        print("          Z =", z, "para n =", n)
        print("          Erro estimado =", eps)
        print("          Erro =", abs(z - integral(h, 1, 2, 10000))/integral(h, 1, 2, 10000))
        print()
        print()
        print("Plot de convergencia de Z, com n = [1:2000] (sem burn-in):")
        plt.plot([(Z(i, S2, "B", 1.42, 0)) for i in range(1,2001)])
        print("     Metodo 3:")
        plt.show()
        print()
        res = find_n(S3, "B", 0.01)
        z = res[0]
        eps = res[1]
        n = res[2]
        print("          Z =", z, "para n =", n)
        print("          Erro estimado =", eps)
        print("          Erro =", abs(z - integral(h, 1, 2, 10000))/integral(h, 1, 2, 10000))
        print()
        print("Plot de convergencia de Z, com n = [1:2000] (sem burn-in):")
        plt.plot([(Z(i, S, "B", 1.42, 0)) for i in range(1,2001)])
        plt.show()
        print()
        print("================================================================================================")
        print()
        print("Plot comparativo entre os alphas para o metodo 1, n = [1:2000] (sem burn-in): ")
        plt.figure, plt.plot([(Z(i, S, "B", 1.42, 0)) for i in range(1,2001)])
        plt.figure, plt.plot([(Z(i, S, "M", 1.42, 0)) for i in range(1,2001)])
        plt.show()
        print()
        print("Plot comparativo entre os alphas para o metodo 1, n = [1:10000] (sem burn-in): ")
        plt.figure, plt.plot([(Z(i, S, "B", 1.42, 0)) for i in range(1,10001)])
        plt.figure, plt.plot([(Z(i, S, "M", 1.42, 0)) for i in range(1,10001)])
        plt.show()
        print()
        res_alt = Z(100000, S, "M", 1.42, 10000)
        res_alt2 = Z(300000, S, "M", 1.42,30000)
        print("Z para uma amostra de tamanho 100000, alfa de Metropolis-Hastings, S = 0.5:", res_alt)
        print("Erro estimado =", abs(res_alt - res_alt2)/res_alt2)
        print("Erro =", abs(res_alt - integral(h, 1, 2, 10000))/integral(h, 1, 2, 10000))
        print()

        stop = 0
        while stop != "s" or stop != "n":
            stop = str(input("Quer recalcular? [s/n] :"))
            if stop == "s":
                break
            elif stop == "n":
                print("---------- ** Programa encerrado ** ----------")
                return
                           
    
main()


##plt.figure, plt.plot(autocorr(MCMC_M(10000, 1.42)))
##plt.show()
##
##t = np.arange(0., 10., 0.01)
##plt.figure, plt.plot(t, g(t))
##x = MCMC_M(10000, 1.42)
##plt.figure, plt.hist(x, density=True, bins=1000)
##plt.show()






    




