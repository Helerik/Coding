import numpy as np
import matplotlib.pyplot as plt

def fun(x):
    return 2*np.exp((x**2 -1)/2)

# Phi(X) leva somente o parametro x, nesse caso. O t esta implicito. Alem disso, X eh o vetor (x,y)
def phi(X, t):
    x = X[0]
    y = X[1]
    return [x*t, y*t]

######################################################################################################

# funcao para qual a raiz precisa ser encontrada no metodo de newton, para metodo Implicito
def X_prox(X, Xk, H, t, t_prox):
    return [X[0] - Xk[0] - H*phi(X, t)[0], X[1] - Xk[1] - H*phi(X, t)[1]]    

# Jacobiana da funcao acima
def JX_prox(X, Xk, H , t, t_prox):
    x = X[0]
    y = X[1]
    return [[1 - H*t, 0], [0, 1 - H*t]]

######################################################################################################

# funcao para qual a raiz precisa ser encontrada no metodo de newton, para metodo do Trapezio
def Y_prox(X, Xk, H, t, t_prox):
    return [X[0] - Xk[0] -(H/2)*(phi(X, t_prox)[0]+phi(Xk, t)[0]), X[1] - Xk[1] -(H/2)*(phi(X, t_prox)[1]+phi(Xk, t)[1])]

# Jacobiana da funcao acima
def JY_prox(X, Xk, H, t, t_prox):
    x = X[0]
    y = X[1]
    xk = Xk[0]
    yk = Xk[1]
    return[[1 - (H/2)*(t_prox + t), 0], [0, 1 - (H/2)*(t_prox + t)]]

######################################################################################################

# Implementacao para o metodo de newton em 2D. Recebe os intervalos para encontrar a raiz para x, Ix e para y, Iy.
# Recebe o X anterior Xk. A funcao e encerrada apos it iteracoes.
def newton_2D(Ix, Iy, Xk, H, it, JX, X_prox, t, t_prox):
    
    xo = Ix[0]
    xf = Ix[1]
    yo = Iy[0]
    yf = Iy[1]
    
    if xf <= xo or yf <= yo:
        return False

    # verifica qual dos extremos dos intervalos deve estar mais proximo da raiz
    raiz = [(xf-xo)/2, (yf-yo)/2]
    raiz_prox = []
    raiz_prox = np.array(raiz) - np.linalg.solve(np.array(JX(raiz, Xk, H, t, t_prox)),np.array(X_prox(raiz, Xk, H, t, t_prox)))

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
        raiz_prox = np.array(raiz) - np.linalg.solve(np.array(JX(raiz, Xk, H, t, t_prox)),np.array(X_prox(raiz, Xk, H, t, t_prox)))
        raiz[:] = raiz_prox
         
    return raiz

# metodo implicito
def euler_imp(Xo, to, tf, n):
    T = [to]
    res = [Xo]
    H = (tf-to)/n

    for _ in range(n):
        X_pro = []
        X_pro = newton_2D([0,200], [0,200], res[-1], H, 10, JX_prox, X_prox, T[-1], T[-1]+H)
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
        X_prox = newton_2D([0,50], [0,50], res[-1], H, 10, JY_prox, Y_prox, T[-1], T[-1]+H)
        res.append(X_prox)
        T.append(T[-1]+H)
    return [T, res]

###############################

# Input dos parametros iniciais

Xo = [2, 2] # xo e yo
to = 1 # tempo inicial
tf = 2 # instante final
n = 6 # numero de iteracoes inicial dos metodos
it = 5 # numero de vezes em que o n sera multiplicado

m = n

# Abaixo estao apenas loops para plottar os graficos

erro_ant = 1

print("Metodo Implicito")
print()
plt.figure()
for i in range(it):
    n *= 2
    eu = euler_imp(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[0] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)

    erro = abs(np.sqrt((eu[-1][0])**2 + (eu[-1][1])**2) - np.sqrt(2*fun(2)**2))
    div_err = erro_ant/erro
    print(round((tf - to)/n,10), " & " ,round(np.sqrt((eu[-1][0])**2 + (eu[-1][1])**2),10), " & ",\
    round(erro, 10), " & ", round(div_err,10), " & ",round(np.log(div_err)/np.log(2),10), "\\\\")
    erro_ant = erro
plt.xlabel("t")
plt.ylabel("X")    
plt.title("Solucao Manufaturada [X - Implicito]")
plt.legend()
plt.grid(True)

print()

n = m
          
plt.figure()
for i in range(it):
    n *= 2
    eu = euler_imp(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[1] for i in eu],label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
plt.xlabel("t")
plt.ylabel("Y")
plt.title("Solucao Manufaturada [Y - Implicito]")
plt.legend()
plt.grid(True)

print()

x,y = eu[-1][0],eu[-1][1]
print(x,y)

print()

n = int(m/3)

erro_ant = 1

print("Metodo do Trapezio")
print()
plt.figure()
for i in range(it):
    n *= 2
    eu = trap(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[0] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
    
    erro = abs(np.sqrt((eu[-1][0])**2 + (eu[-1][1])**2) - np.sqrt(2*fun(2)**2))
    div_err = erro_ant/erro
    print(round((tf - to)/n,10), " & " ,round(np.sqrt((eu[-1][0])**2 + (eu[-1][1])**2),10), " & ",\
    round(erro, 10), " & ", round(div_err,10), " & ",round(np.log(div_err)/np.log(2),10), "\\\\")
    erro_ant = erro
plt.xlabel("t")
plt.ylabel("X")    
plt.title("Solucao Manufaturada [X - Trapezio]")
plt.legend()
plt.grid(True)

print()

n = int(m/3)

plt.figure()
for i in range(it):
    n *= 2
    eu = trap(Xo, to, tf, n)
    T = eu[0]
    eu = eu[1]
    plt.plot(T,[i[1] for i in eu], label = "Dt = %f" %((tf-to)/n), linestyle = (0,(5,5)), lw = 1)
plt.xlabel("t")
plt.ylabel("Y")
plt.title("Solucao Manufaturada [Y - Trapezio]")
plt.legend()
plt.grid(True)

print()

print(eu[-1][0],eu[-1][1])

print()

plt.figure()
u = np.arange(1.,2., 0.01)
plt.plot(u, fun(u), lw = 1)
plt.title("f(u) = 2e^[(u^2 - 1)/2]")
plt.grid(True)

plt.show()
      


    

























