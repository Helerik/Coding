
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def matrixToGraph(mat):
    G = nx.DiGraph()

    for i in range(mat.shape[0]):
        for j in range(i, mat.shape[1]):
            if i == j:
                G.add_node(i+1, name = mat[i][j])
            elif mat[i][j] != 0:
                G.add_edge(i+1,j+1)
                
    labs = {}
    for i in range(len(G.nodes)):
        labs[i+1] = "N%d, [%d]" %(i+1, G.nodes[i+1]['name'])

    nx.draw(G, pos = nx.spring_layout(G, scale = 1, k = 1, iterations = 10, seed = 84), labels = labs, with_labels = True, \
            node_color = 'r', node_size = 1300, font_size = 11, font_weight = 'bold', node_shape = 'o', \
            arrowsize = 0.01, connectionstyle='arc3, rad=0.1')
    plt.pause(0.1)

def isDebt(mat):
    debt = 0
    key = False
    for i in range(mat.shape[0]):
        if mat[i][i] < 0:
            debt += mat[i][i]
            key = True
        if key == True:
            return (True, debt)
    return (False, 0)

def isPlayable(mat):  
    if mat.shape[0] != mat.shape[1]:
        return False
    sum_ = 0
    for i in range(mat.shape[0]):
        hasCon = 0
        for j in range(mat.shape[1]):
            if i == j:
                sum_ += mat[i][j]
            else:
                hasCon += mat[i][j]
        if hasCon == 0:
            return False
    if sum_ < 0:
        return False
    return True

def moveDonate(node, mat):
    conecs = 0
    for i in range(mat.shape[1]):
        if node-1 == i:
            pass
        elif mat[node-1][i] == 1:
            conecs += 1
            mat[i][i] += 1
    mat[node-1][node-1] -= conecs
    return mat

def moveTake(node, mat):
    conecs = 0
    for i in range(mat.shape[1]):
        if node-1 == i:
            pass
        elif mat[node-1][i] == 1:
            conecs += 1
            mat[i][i] -= 1
    mat[node-1][node-1] += conecs
    return mat

def DollarGame(mat):
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            mat[j][i] = mat[i][j]
            
    if not isPlayable(mat):
        print("This graph is unplayable! Use a different graph to play!")
        return
    debt = isDebt(mat)
    move_count = 0
    while debt[0] == True:
        move = None
        while move != 3:
            move = None
            plt.cla()
            matrixToGraph(mat)
            
            print("Choose a node:")
            nod = ""
            while not isinstance(nod, int) or nod not in range(1, mat.shape[1]+1):
                try:
                    nod = int(input())
                except:
                    pass
            print("Chosen node: %d \n" %nod)
            
            print("Choose a move:\n")
            print("[1 - Donate]\n[2 - Take]\n[3 - Choose another node]\n[4 - Give up]\n")
            move = ""
            while not isinstance(move, int) or not move in range(1, 5):
                try:
                    move = int(input())
                except:
                    pass
                
            if move == 1:
                print("You chose to donate.")
                print("-"*80 + "\n")
                mat = moveDonate(nod, mat)
                debt = isDebt(mat)
                move_count += 1
                
            elif move == 2:
                print("You chose to take.")
                print("-"*80 + "\n")
                mat = moveTake(nod, mat)
                debt = isDebt(mat)
                move_count += 1

            elif move == 3:
                pass
            elif move == 4:
                print("You chose to give up. Good luck next time!")
                print("Total debt: %d" %debt[1])
                print("Toal moves: %d" %move_count)
                return
            
            if debt[0] == False:
                plt.cla()
                matrixToGraph(mat)
                break

    print("-"*80)
    print()
    print("All debt is clear. You won in %d moves." %move_count)
    print("Congratulations!")
    print()
    print("-"*80)
    print()
    return
        



n = 8
mat = np.random.randint(low = -4, high = 5, size = (n,n))
mat = np.triu(mat,0)
for i in range(mat.shape[0]):
    for j in range(i, mat.shape[1]):
        if i == j:
            pass
        elif i != j:
            if mat[i][j] <= 0:
                mat[i][j] = 0
            else:
                mat[i][j] = 1
print(mat)
print()

##mat = np.array([
##    [-2,0,1,0,1],
##    [0,1,0,0,1],
##    [0,0,2,1,0],
##    [0,0,0,-1,1],
##    [0,0,0,0,2]
##    ])

DollarGame(mat)







