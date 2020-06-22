import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def OptObjctv(X, mu, C):
    sum_ = 0
    for i in range(len(X)):
        sum_ += np.linalg.norm(X[i] - mu[int(C[i])])**2
    return sum_/len(X)

def indexPoint(mu, X, xIndex):
    x = X[xIndex]
    comparissons = [np.linalg.norm(x - mu[i])**2 for i in range(len(mu))]
    return comparissons.index(np.min(comparissons))

def K_Means(mu, X, iters, graphics = 0, delete_mus = 0, graphs_time = 0.5):
    
    colr = list(colors._colors_full_map.values())
    np.random.shuffle(colr)
    time = graphs_time
    C = np.zeros(X.shape[0])
    if graphics == 1:
        plt.figure(1)
        
    for j in range(iters):
        
        if graphics == 1:
            plt.plot(-2,-2, color = (0,0,0,0), marker = 'o', linestyle = '')
            plt.plot(22,22, color = (0,0,0,0), marker = 'o', linestyle = '')
            plt.plot(X[:,0], X[:,1], 'o', color = "black")
            for i in range(len(mu)):
                plt.plot(mu[i:,0], mu[i:,1], marker = 'x', color = colr[i], linestyle = '')

            plt.pause(time)
        
        for i in range(len(X)):
            C[i] = indexPoint(mu, X, i)

        sub_k = 0
        for k in range(len(mu)):
            lis = np.where(C == k)[0]
            if lis.size == 0:
                if delete_mus == 1:
                    mu = np.delete(mu,k - sub_k,0)
                    del(colr[k - sub_k])
                    sub_k += 1
                continue
            vals = np.asarray([X[i] for i in lis])
            mu[k - sub_k] = np.mean(vals, axis = 0)

            if graphics == 1:
                plt.plot(vals[:,0], vals[:,1], marker = 'o', color = colr[k - sub_k], linestyle = '')
                plt.pause(time)
        if graphics == 1 and j != iters - 1:
            plt.clf()
            plt.cla()
    return [C, mu]

##mu = 20*np.random.rand(4,2)
X = np.array([
    [2.5,3.4],
    [1.5,6.9],
    [1.1,1],
    [0,3.7],
    [0,0.5],
    [9,10.1],
    [9,11],
    [8.2,10],
    [20.3,4],
    [15.5,13],
    [12.2,16],
    [19,18.1],
    [18,18.3],
    [15.5,17],
    [3,4.5],
    [4.5,3],
    [6,5.6],
    [7.8,2],
    [8.9,9],
    [2.3,8],
    [1.2,9.9],
    [3.3,7.6],
    [7.6,19.9],
    [18,3.8],
    [-2,-2],
    [-1,-2],
    [-1.5,-1],
    [-1.9,2],
    [-0.6,-1],
    [-1.1,-1.1],
    [-0.9,-0.8],
    [-0.9,-1.2],
    [-0.1,0]
    ])


opt = []
for j in range(4):
    mu_size = j + 1
    min_ = []
    for i in range(10):
        randRows = np.random.randint(len(X), size = mu_size)
        mu = np.zeros((mu_size, 2))
        for i in range(mu_size):
            mu[i] = X[randRows[i]]
        res = K_Means(mu, X, 10, graphics = 1, delete_mus = 1, graphs_time = 0.05)

        min_.append(OptObjctv(X, res[1], res[0]))

##        print("C =\n", res[0])
##        print()
##        print("Mus =\n", res[1])
##        print()
##        print("Optimization Objective =\n", min_[-1])
##        print()
    
    plt.figure(2)  
    opt.append(min(min_))
    plt.plot(opt)
    plt.pause(0.1)

