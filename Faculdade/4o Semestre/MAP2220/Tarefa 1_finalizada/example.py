import numpy as np # Biblioteca para lidar com objetos N-dimensionais
import matplotlib.pyplot as plt # Biblioteca grafica
import math #Bliblioteca com funcoes matematicas

# Phi de um metodo de primeira ordem (Euler)
def euler(t, x, h, f):
  return f( t, x )

# Definicao da funcao f

# Inicialmente definimos as variaveis auxiliares
A = np.array([[-1,0],[0,-2]],dtype=float)

# Declaracao da funcao f(t,x) = Ax
def func_linear( t, x ):
  # Multiplicacao matricial
  return np.matmul(A,x)

x0 = np.array([1,1]) # Condicao Inicial
t0 = 0 # Instante de tempo inicial
tf = 5 # Instante de tempo final
n = 1e3 # Número de passos

def ode_explicit( t_start, t_end, x_start, steps, f, phi ):
  # Inicializacao de variaveis
  t = np.linspace(t_start,t_end,int(steps+1),endpoint=True) # t_i's
  x = np.zeros([len(x_start),int(steps+1)]) # x_i's
  
  h = (t_end-t_start)/steps

  x[:,0] = x_start # Copia condicao inicial
  for i in range(int(steps)):
    x[:,i+1] = x[:,i] + h*phi(t[i],x[:,i],h,f)
    
  return t,x

t,x = ode_explicit( t0, tf, x0, steps = n, f = func_linear, phi = euler )
print(x)

plt.figure(figsize=(10,5)) # Modificacao do tamanho do grafico

# Plotar as componentes de x
for i in range(x.shape[0]):
    plt.plot(t, x[i,:], label = 'dimensao ' + str(i+1))
  
# Rotulos dos eixos
plt.xlabel('t')
plt.ylabel('x')

# Titulo do grafico
plt.title('Método de Euler (n = %d)'%(n))

# Alterar limites inferior e superior do eixo x (use ylim para eixo y)
plt.xlim([ t0, tf ])

plt.grid(True) # Mostrar linhas de grade
plt.legend() # Mostrar legenda
plt.show() # Mostrar grafico









