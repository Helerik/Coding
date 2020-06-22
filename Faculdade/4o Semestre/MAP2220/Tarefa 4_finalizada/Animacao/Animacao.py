# -*- coding: utf-8 -*- #

# Funcao de Animacao

# Erik Davino Vincent; NUSP: 10736583

from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# Transforma o txt em lista (opcional)
def txt_to_list(txt):
    f = open(txt, "r")
    ret = []
    data = f.readlines()
    for line in data:
        words = line.split()
        ret.append([float(word) for word in words])
    return ret

# Funcao de animacao generalizada (VEJA O FIM DO ARQUIVO)
def animate(data, minX, maxX, minY, maxY, titulo, start_end = True, grid = True, tposx = 0.5, tposy = 0.8, adt_info = False, spd = 500, FPS):
    
    Writer = writers['ffmpeg']
    writer = Writer(fps=FPS, metadata=dict(artist=''), bitrate=1800)

    def init():
        ln.set_data(xdata,ydata)
        return ln,
        
    def update(frame):
        
        if key == 1:
            xdata.append(data[frame][0])
            ydata.append(data[frame][1])
        else:
            res = []
            file = open(data)
            info = [file.readlines()[frame]]
            info = info[0].split()
            res.append(info)
            xdata.append(int(info[0]))
            ydata.append(int(info[1]))
            file.close()
            
        if adt_info != False and isinstance(adt_info, str):
            title.set_text(adt_info)
        elif adt_info != False and isinstance(adt_info, list):
            title.set_text(adt_info[frame])
        
        ln.set_data(xdata, ydata)
        
        if adt_info != False:
            return ln, title
        else:
            return ln,
    
    if isinstance(data, list):
        key = 1
    elif isinstance(data, str):
        key = 0
        try:
            with open(data) as f:
                size_data = 0
                for line in f:
                    size_data += 1
        except:
            print("Data deve ser <nome_do_arquivo>.txt ou lista.")
            exit()
    else:
        print("Data deve ser <nome_do_arquivo>.txt ou lista.")
        exit()
        
    fig, ax = plt.subplots()

    if grid == True:
        plt.grid()
    
    plt.xlim(minX, maxX)
    plt.ylim(minY, maxY)
    
    plt.title(titulo)
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")

    if start_end == True:
        if key == 1:
            plt.plot(data[0][0], data[0][1], 'bo', label = "inicio")
            plt.plot(data[-1][0], data[-1][1], 'ro', label = "fim")
        elif key == 0:
            res = []
            file = open(data)
            info = [file.readlines()[0]]
            info = info[0].split()
            res.append(info)
            file.close()
            file = open(data)
            info = [file.readlines()[-1]]
            info = info[0].split()
            res.append(info)
            plt.plot(int(res[0][0]), int(res[0][1]), 'bo', label = "inicio")
            plt.plot(int(res[-1][0]), int(res[-1][1]), 'ro', label = "fim")
            file.close()
                

    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r', animated=True, linestyle = (0, (1, 1)))
    if key == 1:
        f = np.array([i for i in range(len(data))])
    else:
        f = np.array([i for i in range(size_data)])

    if adt_info != False:
        title = ax.text(tposx,tposy, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                        transform=ax.transAxes, ha="center")
    plt.legend()
    ani = FuncAnimation(fig, update, frames=f, init_func=init, blit=True, interval = 500 ,repeat=False)
    ani.save("MyAnimation.mp4", writer = writer)

###########################################################################################################################
''' Provavelmente nao sera necessario alterar nada alem dos valores dentre os parametros abaixo '''

# Parametros:

data = [[0,1],[1,2],[2,4],[3,3],[4,2]] # exemplo de plot (sua matriz de pontos [x, y])
''' Alternativamente, data pode ser <nome_do_arquivo>.txt, desde que esteja configurado como a seguir:
    0 1
    1 2
    2 4
    3 3
    4 2
    O exemplo acima em um txt deve executar exatamente o mesmo que a lista acima'''
# As quatro coordenadas abaixo definem o 'pedaco' do grafico que sera apresentado
minX = -1 
minY = -1
maxX = 5
maxY = 5

titulo = "titulo" # titulo do grafico

start_end = True # Se "True" mostra o inicio e o fim da trajetoria, c.c, nao.

grid = True # Utiliza uma "malha" ou nao.

# Os dois valores abaixo sao coordenadas de uma caixa de texto para informacoes adicionais (A caixa de texto eh ativada
# somente se "adt_info for do tipo str")
tposx = 0.5
tposy = 0.9

adt_info = "algo algo algo" # apresenta o texto ao lado na caixa de texto. Caso queira desativado, deixar como "False"
''' adt_info pode ser uma lista de str's, de forma que pode ser atualizado a cada frame, por exemplo o tempo'''
adt_info = ["Tempo = %d segundos" %i for i in range(5)]

spd = 500 # A velocidade com que cada frame eh passado

fps = 2 # frames por segundo da animacao final

###########################################################################################################################

# Executa a animacao:
animate(data, minX, maxX, minY, maxY, titulo, start_end, grid, tposx, tposy, adt_info, spd, fps)









            
