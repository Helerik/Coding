import numpy as np
import matplotlib.pyplot as plt
import time

# Parametros da phi(X)
al = 0.87
be = 0.27
ga = 0.38
de = 0.25

# Phi(X) leva somente o parametro x, nesse caso. O t esta implicito. Alem disso, X eh o vetor (x,y)
def phi(X):
    x = X[0]
    y = X[1]
    return [al*x - be*x*y, -ga*y + de*y*x]

def Hphi(X, H):
    x = X[0]
    y = X[1]
    return [H*(al*x - be*x*y), H*(-ga*y + de*y*x)]

######################################################################################################

# funcao para qual a raiz precisa ser encontrada no metodo de newton, para metodo Implicito
def X_prox(X, Xk, H):
    v = X[0] - Xk[0] - H*phi(X)[0]
    return [X[0] - Xk[0] - H*phi(X)[0], X[1] - Xk[1] - H*phi(X)[1]]

# Jacobiana da funcao acima
def JX_prox(X, Xk, H):
    x = X[0]
    y = X[1]
    return [[1 - H*(al-be*y),-H*(-be*x)], [-H*(de*y),1 - H*(-ga+de*x)]]

######################################################################################################

# funcao para qual a raiz precisa ser encontrada no metodo de newton, para metodo do Trapezio
def Y_prox(X, Xk, H):
    return [X[0] - Xk[0] -(H/2)*(phi(X)[0]+phi(Xk)[0]),X[1] - Xk[1] -(H/2)*(phi(X)[1]+phi(Xk)[1])]

# Jacobiana da funcao acima
def JY_prox(X, Xk, H):
    x = X[0]
    y = X[1]
    xk = Xk[0]
    yk = Xk[1]
    return[[1 - (H/2)*(al-be*y + al-be*yk), -(H/2)*(-be*x - be*xk)], [-(H/2)*(de*y + de*yk), 1 - (H/2)*(-ga + de*x -ga + de*xk)]]

######################################################################################################

# Implementacao para o metodo de newton em 2D. Recebe os intervalos para encontrar a raiz para x, Ix e para y, Iy.
# Recebe o X anterior Xk. A funcao e encerrada apos it iteracoes.
def newton_2D(Ix, Iy, Xk, H, it, JX, X_prox):
    
    xo = Ix[0]
    xf = Ix[1]
    yo = Iy[0]
    yf = Iy[1]
    
    if xf <= xo or yf <= yo:
        return False

    # verifica qual dos extremos dos intervalos deve estar mais proximo da raiz
    raiz = [(xf-xo)/2, (yf-yo)/2]
    raiz_prox = []
    raiz_prox = np.array(raiz) - np.linalg.solve(np.array(JX(raiz, Xk, H)),np.array(X_prox(raiz, Xk, H)))

    if raiz_prox[0] < raiz[0]:
        raiz[0] = xo
    else:
        raiz[0] = xf
    if raiz_prox[1] < raiz[1]:
        raiz[1] = yo
    else:
        raiz[1] = yf

    # loop do metodo. Utilizando o np.linalg.solve aceleramos o programa e evitamos problemas no momento de inverter a jacobiana (erro de matriz singular)
    for _ in range(it):
        raiz_prox = np.array(raiz) - np.linalg.solve(np.array(JX(raiz, Xk, H)),np.array(X_prox(raiz, Xk, H)))
        raiz[:] = raiz_prox
         
    return raiz

# metodo implicito
def euler_imp(Xo, to, tf, n):
    T = [to]
    res = [Xo]
    H = (tf-to)/n

    for _ in range(n):
        X_pro = []
        X_pro = newton_2D([0,200], [0,200], res[-1], H, 10, JX_prox, X_prox)
        res.append(X_pro)
        T.append(T[-1]+H)
    return [T, res]

# metodo do trapezio
def trap(Xo, to, tf, n):
    T = [to]
    res = [Xo]
    H = (tf-to)/n

    for _ in range(n):
        X_prox = []
        X_prox = newton_2D([0,50], [0,50], res[-1], H, 10, JY_prox, Y_prox)
        res.append(X_prox)
        T.append(T[-1]+H)
    return [T, res]
    
# metodo de euler, para comparacao
def euler(Xo, to, tf, n):
    T = [to]
    res = [Xo]
    H = (tf-to)/n

    for _ in range(n):
        X_prox = []
        X_prox = np.array(res[-1]) + np.array(Hphi(res[-1], H))
        res.append(X_prox)
        T.append(T[-1]+H)

    return [T, res]

###############################

# Input dos parametros iniciais

Xo = [3.5, 2.7] # xo e yo
to = 0 # tempo inicial
tf = 60 # instante final
n = 300 # numero de iteracoes inicial dos metodos
it = 6 # numero de vezes em que o n sera multiplicado

m = n


# Abaixo estao apenas loops para plottar os graficos

print("Metodo Implicito")
print()
plt.figure()
for i in range(it):
    t = time.time()
    print(i+1, end = " - Tempo: ")
    n *= 2
    eu = euler_imp(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[0] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
    print(time.time()-t)
plt.xlabel("t")
plt.ylabel("Presa")
plt.title("Presa-Predador de Lotka-Volterra [X - Implicito]")
plt.legend()
plt.grid(True)

print()

n = m
          
plt.figure()
for i in range(it):
    t = time.time()
    print(i+1, end = " - Tempo: ")
    n *= 2
    eu = euler_imp(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[1] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
    print(time.time()-t)
plt.xlabel("t")
plt.ylabel("Predador")
plt.title("Presa-Predador de Lotka-Volterra [Y - Implicito]")
plt.legend()
plt.grid(True)

print()

x,y = eu[-1][0],eu[-1][1]
print(x,y)

print()

n = int(m/3)

print("Metodo do Trapezio")
print()
plt.figure()
for i in range(it):
    t = time.time()
    print(i+1, end = " - Tempo: ")
    n *= 2
    eu = trap(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[0] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
    print(time.time()-t)
plt.xlabel("t")
plt.ylabel("Presa")
plt.title("Presa-Predador de Lotka-Volterra [X - Trapezio]")
plt.legend()
plt.grid(True)

print()

n = int(m/3)

plt.figure()
for i in range(it):
    t = time.time()
    print(i+1, end = " - Tempo: ")
    n *= 2
    eu = trap(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[1] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
    print(time.time()-t)
plt.xlabel("t")
plt.ylabel("Predador")
plt.title("Presa-Predador de Lotka-Volterra [Y - Trapezio]")
plt.legend()
plt.grid(True)

print()

print(eu[-1][0],eu[-1][1])

print()

n = m

print("Metodo Explicito")
print()
plt.figure()
for i in range(it):
    t = time.time()
    print(i+1, end = " - Tempo: ")
    n *= 2
    eu = euler(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[0] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
    print(time.time()-t)
plt.xlabel("t")
plt.ylabel("Presa")
plt.title("Presa-Predador de Lotka-Volterra [X - Euler]")
plt.legend()
plt.grid(True)

print()

n = m

plt.figure()
for i in range(it):
    t = time.time()
    print(i+1, end = " - Tempo: ")
    n *= 2
    eu = euler(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[1] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
    print(time.time()-t)
plt.xlabel("t")
plt.ylabel("Predador")
plt.title("Presa-Predador de Lotka-Volterra [Y - Euler]")
plt.legend()
plt.grid(True)

print()

print(eu[-1][0],eu[-1][1])

print()

print("Media explicito com implicito = ", (eu[-1][0] + x)/2,(eu[-1][1] + y)/2)


plt.show()
    
        

      


    

























