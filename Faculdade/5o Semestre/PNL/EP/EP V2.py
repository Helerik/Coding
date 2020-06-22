import numpy as np
import matplotlib.pyplot as plt
import time

# Objective function to be minimized wih respect to x
def f(x):
    retf = 0
    for i in range(1, len(x)):
        retf += x[i]*dx
    return retf

# Gradient of the objective function
def Grad_f(x):
    retgrad = []
    for i in range(len(x)):
        if i == 0:
            grad = dx
        elif i == len(x)-1:
            grad = dx
        else:
            grad = dx
        retgrad.append(grad)
    return np.array(retgrad)

# Hessian of the objective function
def Hess_f(x):
    
    rethess = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        if i == 0:
            rethess[i][i] = 0
        elif i == len(x)-1:
            rethess[i][i] = 0
        else:
            rethess[i][i] = 0
    for i in range(len(x)-1):
        rethess[i][i+1] = 0
        rethess[i+1][i] = 0
                
    return rethess

# Equality constraints: returns a vector of equality constraint functions
def h(x):
    all_h = []

    h1 = x[0]
    h2 = x[len(x)-1]
    h3 = 0
    h4 = x[len(x)//2]
    for i in range(1, len(x)):
        h3 += dx**2 + (x[i] - x[i-1])**2 - (3/(len(x)-1))**2

    all_h.append(h1)
    all_h.append(h2)
    all_h.append(h3)
##    all_h.append(h4)
    
    return np.array(all_h)

# Equality constraints jacobian: returns a matrix; each line of the matrix is the gradient of each constraint function
def Jacob_h(x):
    all_grad = []
    
    grad_h1 = [1]
    for i in range(1, len(x)):
        grad_h1.append(0)
    grad_h2 = [0 for i in range(len(x)-1)]
    grad_h2.append(1)
    grad_h3 = []
    grad_h4 = [0 for i in range(len(x))]
    grad_h4[len(x)//2] = 1
    for i in range(len(x)):
        if i == 0:
            grad_h3.append(-2*(x[i+1] - x[i]))
        elif i == len(x)-1:
            grad_h3.append( 2*(x[i] - x[i-1]))
        else:
            grad_h3.append(-2*(x[i-1] - 2*x[i] + x[i+1]))

    all_grad.append(grad_h1)
    all_grad.append(grad_h2)
    all_grad.append(grad_h3)
##    all_grad.append(grad_h4)
    
    return np.array(all_grad)

# Returns a vector containing the hessian of each constraint function h
def Hess_h(x):
    all_hess = []

    hess_h1 = np.zeros((len(x), len(x)))
    hess_h2 = np.zeros((len(x), len(x)))
    hess_h3 = np.zeros((len(x), len(x)))
    hess_h4 = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        if i == 0:
            hess_h3[i][i] = 2
        elif i == len(x)-1:
            hess_h3[i][i] = 2
        else:
            hess_h3[i][i] = 4
    for i in range(len(x)-1):
        hess_h3[i][i+1] = -2
        hess_h3[i+1][i] = -2

    all_hess.append(hess_h1)
    all_hess.append(hess_h2)
    all_hess.append(hess_h3)
##    all_hess.append(hess_h4)

    return np.array(all_hess)
    
# Gradient of the lagrangian, with respect to x
def Grad_Lag(x, lamb):
    JH = Jacob_h(x)
    sum_jacob_h = np.zeros(JH[0].shape)
    for i in range(len(lamb)):
        sum_jacob_h = sum_jacob_h + lamb[i]*JH[i]

    return Grad_f(x) + sum_jacob_h

# Hessian of the lagrangian, with respect to x
def Hess_Lag(x, lamb):
    HH = Hess_h(x)
    sum_hess_h = np.zeros((HH[0].shape))
    for i in range(len(lamb)):
        sum_hess_h = sum_hess_h + lamb[i]*HH[i]

    return Hess_f(x) + sum_hess_h

# Verifies if a matrix is positive definite
def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)

# Creates a positive definite aproximation to a not positive definite matrix
def aprox_pos_def(A):
    u, V = np.linalg.eig(A)
    U = np.diag(u)
    for i in range(len(U)):
        for j in range(len(U)):
            if U[i][j] < 0:
                U[i][j] = -U[i][j]
    B = np.dot(V, np.dot(U, V.T))
    
    return B

# Creates matrix used in the system solving step of the newton method
def make_step_matrix(x, lamb):
    matrix = Hess_Lag(x, lamb)

    # If the matrix is not positive definite, we can use an aproximation that is
    if not is_pos_def(matrix):
        matrix = aprox_pos_def(matrix)

    vec = Jacob_h(x)
    
    matrix = np.append(matrix, vec.T, axis = 1)
    
    vec = np.append(vec, np.zeros((vec.shape[0], vec.shape[0])), axis = 1)

    matrix = np.append(matrix, vec, axis = 0)

    return matrix

# Merit function
def mer(x, mu):
    return f(x) + mu*np.linalg.norm(h(x))

# Derivative of the merit function with respect to x
def D_mer(x, mu, d):
    return np.dot(Grad_f(x).T, d) - mu*np.linalg.norm(h(x))

# Armijo line search for the merit function
def armijo_search_mer(x, mu, d, N = 4/5, gamma = 0.8):
    t = 1
    while mer(x + t*d, mu) > mer(x, mu) + N*t*D_mer(x,mu,d):
        t = t*gamma
    return t

# Finds a direction dv and takes the step in that direction for x and lambda
def newton_step(x_k, lamb_k, mu_k, ro):

    b = Grad_Lag(x_k, lamb_k)
    b = np.append(b, h(x_k))

    dv = np.linalg.solve(make_step_matrix(x_k, lamb_k), -b.T)
    dvx = dv[:len(x_k)]
    dvl = dv[len(x_k):]
    
    cond = (np.dot(Grad_f(x_k), dvx) + 0.5*np.dot(dvx.T, np.dot(Hess_Lag(x_k, lamb_k), dvx)))/((1-ro)*np.linalg.norm(h(x_k)))
    try:
        if mu_k < abs(cond):
            mu_k = abs(cond)*1.01
    except:
        mu_k = abs(cond)*1.01

    t = armijo_search_mer(x_k, mu_k, dvx)

    dvx = dv[:len(x_k)]
    dvl = dv[len(x_k):]
    
    lamb_k = lamb_k + t*dvl
    x_k = x_k + t*dvx

    return (x_k, lamb_k, mu_k)

# Function optimizer using Newton Method:
# x_0: initial guess for x (array)
# lamb_0: initial guess for lambda (array)
# max_iters: maximum number of iterations (int)
# tol: tolerance for concluding convergence (float)
# verbose: the higher the value, the more messages will appear (int in [0,2])
# do_plot: if 1, plots x (int in [0,1])
# ro: constant for determining size of penalty. May affect convergence speed (float in (0,1))
def newton_opt(x_0, lamb_0, max_iters = 200, tol = 1e-10, verbose = 1, do_plot = 0, ro = 0.9):
    t = time.time()
    
    x_k = x_0
    lamb_k = lamb_0
    mu_k = None
    
    if do_plot == 1:
        plt.figure()

    if verbose > 1:
        print("Iteration 0")
        print()
        print("Initial guess for x: ", end = "(")
        for i in range(len(x_0)):
            if i == len(x_0)-1:
                print(round(x_0[i], 4), end = '')
                break
            print(round(x_0[i], 4), end = ', ')
        print(")")
        print("f(x0) =", f(x_0))

    for _ in range(max_iters):
        x_cur = x_k
        lamb_cur = lamb_k
        try:
            x_k, lamb_k, mu_k = newton_step(x_k, lamb_k, mu_k, ro)
        except:
            try:
                rand_sign = 1
                if np.random.random() < 0.5:
                    rand_sign = -1
                x_k = x_k + rand_sign*np.random.random(2)
                x_k, lamb_k = newton_step(x_k, lamb_k)
            except:
                print()
                print("* An error ocurred *")
                print()
                return (x_k, lamb_k)
            
        if do_plot == 1:
            plt.clf()
            plt.plot([i*dx for i in range(len(x_k))], x_k)
            plt.scatter([i*dx for i in range(len(x_k))], x_k)
            plt.grid()
            plt.pause(0.001)
            
        ret_key = 1
        ret_eval = Grad_f(x_k)
        for val in ret_eval:
            if abs(val) > tol:
                ret_key = 0
        if ret_key == 1:
            return (x_k, lamb_k)

        if np.max(Grad_Lag(x_k, lamb_k)) < tol:
            if verbose > 0:
                print()
                print("Convergence achieved in", _, "iterations.")
                print("Elapsed time:", time.time() - t)
                print()
                plt.show()
            return (x_k, lamb_k)

        if verbose > 1:
            print()
            print("Iteration", _+1)
            print()
            print("Current best solution: ", end = "(")
            for i in range(len(x_k)):
                if i == len(x_k)-1:
                    print(round(x_k[i], 4), end = '')
                    break
                print(round(x_k[i], 4), end = ', ')
            print(")")
            print("Current lambda: ", end = "(")
            for i in range(len(lamb_k)):
                if i == len(lamb_k)-1:
                    print(round(lamb_k[i], 4), end = '')
                    break
                print(round(lamb_k[i], 4), end = ', ')
            print(")")
            print("f(xk) =", f(x_k))

    if verbose > 0:
        print()
        print("Max number of iterations reached (%d)" %max_iters)
        print("Elapsed time:", time.time() - t)
        print()

    plt.show()
    
    return (x_k, lamb_k)

# number of points to approximate x(t)
n = 101

# initial guess for y(x)
x_init = np.random.uniform(-10,10,n)

# initial guess for lambda
lamb_init = np.random.uniform(-1, 1, 3)

# constant of derivation/integration
dx = 2/(len(x_init)-1)

# execution of optimization algorithm based on newton step
x, lamb = newton_opt(x_init, lamb_init, max_iters = 1000, verbose = 1, do_plot = 1, ro = 0.9, tol = 1e-5)

# print final results
print()
print("Final results:")
print("f(x) =", f(x))
print("Lambda:", lamb)
print()

# plot graph
t = np.arange(0., 2. + dx, dx)
plt.figure()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Numerical solution for y(x)")
plt.plot(t, x)
plt.scatter(t, x)
plt.grid()
plt.show()






























