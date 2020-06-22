import matplotlib.pyplot as plt
import numpy as np

# função de "passo", dx/dt
def f(t,x):
    return np.exp(t)*(np.sin(2*t) + 2 * np.cos(2*t))

def f2(t,x):
    return x

# metodo de EULER
def euler(to, tf, n, xo, phi):
    H = (tf-to)/n
    
    vec_x = [xo] # vetor com cada ponto obtido de x
    vec_t = [to] # vetor com cada ponto de t

    for i in range(1, n+1, 1):
        vec_x.append(vec_x[i-1] + H*phi(vec_t[i-1],vec_x[i-1]))
        vec_t.append(vec_t[i-1] + H)

    return (vec_t, vec_x)

# metodo de Euler Modificado
def euler_mod(to, tf, n, xo, phi):
    H = (tf-to)/n
    
    vec_x = [xo] # vetor com cada ponto obtido de x
    vec_t = [to] # vetor com cada ponto de t

    for i in range(1, n+1, 1):
        tmp = phi(vec_t[i-1],vec_x[i-1]) # pequena otimização, para calcular a phi apenas uma vez
        vec_x.append(vec_x[i-1] + (H/2)*(tmp + phi(vec_t[i-1] + H, vec_x[i-1] + H*tmp)))
        vec_t.append(vec_t[i-1] + H)

    return (vec_t, vec_x)

def main():

    # variaveis de entrada:
    xo = 1 # x(to) inicial
    to = 0 # t inicial
    tf = 2 # t final
    n = 1 # n inicial/L
    L = 2 # fator de aumento de n
    fun = f # escolha da funcao (f ou f2)
    metodo = euler # escolha do metodo (euler ou euler_mod)

    # plotagem da função verdadeira de f (nao de f2, pode ser ignorado se necessario)
    plt.figure(figsize = (7,5))
    u = np.arange(0.,2.,0.01)
    plt.plot(u, np.exp(u)*np.sin(2*u)+1, label = "x(t)")

    
    erro_ant = 1 #  erro anterior inicial (que não existe, mas é necessário para o cálculo não falhar)
    for k in range(10):
        n *= L
        eu = metodo(to, tf, n, xo, fun) #chama o metodo escolhido
        plt.plot(eu[0], eu[1], lw = 1, dashes = [6,2], label = ("dt = %f" %((tf-to)/n))) # plot da função pelo método
        
        erro = abs(eu[1][-1] - 3.471726672)
        div_err = erro_ant/erro
        print(round((tf - to)/n,10), " & " ,round(eu[1][-1],10), " & ",\
        round(erro, 10), " & ", round(div_err,10), " & ",round(np.log(div_err)/np.log(L),10), "\\\\")
        erro_ant = erro

    # configuração do plot
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.title("Plot da convergencia")
    plt.show()

main()





