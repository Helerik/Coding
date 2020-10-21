# !/usr/bin/env python3
# Author: Erik Davino Vincent

import time

class Sparse():

    def __init__(self, matrix = []):
        self.sparse = []
        
        if isinstance(matrix, list) and matrix:
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if matrix[i][j] != 0:
                        self.sparse.append((matrix[i][j], i, j))
            self.sparse.append(("", len(matrix),len(matrix[0])))

    def transpose(self):
        ret = []
        for i in range(len(self.sparse)-1):
            ret.append((self.sparse[i][0], self.sparse[i][2], self.sparse[i][1]))
        ret.append(("", self.sparse[-1][2], self.sparse[-1][1]))
        ret_val = Sparse()
        ret_val.sparse = ret
        return ret_val

    def unsparse(self):
        ret = [[0 for j in range(self.sparse[-1][2])] for i in range(self.sparse[-1][1])]
        for i in range(len(self.sparse)-1):
            v, i, j = self.sparse[i]
            ret[i][j] = v
        return ret

    def sum(self, other):
        mtx1 = self.sparse.copy()
        mtx2 = other.sparse.copy()
        dims = mtx1.pop(-1)
        mtx2.pop(-1)

        for i in range(len(mtx1)):
            key = 0
            for j in range(len(mtx2)):
                if mtx1[i][1] == mtx2[j][1] and mtx1[i][2] == mtx2[j][2]:
                    mtx1[i] = (mtx1[i][0] + mtx2[j][0], mtx1[i][1], mtx1[i][2])
                    mtx2.pop(j)
                    break
        for j in range(len(mtx2)):
            mtx1.append(mtx2[j])

        mtx1.append(dims)
        ret_val = Sparse()
        ret_val.sparse = mtx1
        return ret_val

    def multiply(self, other):
        mtx1 = self.sparse.copy()
        mtx2 = other.sparse.copy()
        dims = mtx1.pop(-1)
        mtx2.pop(-1)

        for i in range(len(mtx1)):
            key = 0
            for j in range(len(mtx2)):
                if mtx1[i][1] == mtx2[j][1] and mtx1[i][2] == mtx2[j][2]:
                    mtx1[i] = (mtx1[i][0] * mtx2[j][0], mtx1[i][1], mtx1[i][2])
                    mtx2.pop(j)
                    break
        for j in range(len(mtx2)):
            mtx1.append(mtx2[j])

        mtx1.append(dims)
        ret_val = Sparse()
        ret_val.sparse = mtx1
        return ret_val

mat1 = [[1,0,0,0,0],[4,0,0,0,8],[7,0,0,0,11],[0,0,0,0,0],[13,0,15,0,17],[13,0,0,0,17],[13,0,0,0,17],
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

mat2 = [[0,0,0,4,0],[1,0,0,4,0],[0,0,0,0,5],[0,0,0,0,5],[1,0,0,0,5],[13,0,0,0,17],[13,0,0,0,17],
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],
        [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

mat = [[0 for i in range(len(mat1))] for j in range(len(mat1))]

t = time.time()
for _ in range(10000):
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            mat[i][j] = mat1[i][j] + mat2[i][j]
print(f"Avrg time: {(time.time() - t)/1000}")

s1 = Sparse(mat1)
s2 = Sparse(mat2)

t = time.time()
for _ in range(10000):
    s1.sum(s2)
print(f"Avrg time: {(time.time() - t)/1000}")


    



        
            
