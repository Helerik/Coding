# !/usr/bin/env python3
# Author: Erik Davino Vincent

class Sparse():

    def __init__(self, matrix = []):

        self.sparse = []
        if isinstance(matrix, list):
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if matrix[i][j] != 0:
                        self.sparse.append((matrix[i][j], i, j))
        self.sparse.append(("", len(matrix),len(matrix[0]))

    def transpose(self):
        ret = []
        for i in range(len(self.sparse)-1):
            ret.append(self.sparse[i][0], self.sparse[i][2], self.sparse[i][1])
        ret.append("", self.sparse[-1][2], self.sparse[-1][1])
        return ret

    def multiply(self, other):
        mtx1 = self.sparse
        mtx2 = other.sparse

        res = []
        for i in range(len(mtx1)):
            sum_ = 0
            for j in range(len(mtx2)):
                if mtx1[i][1] == mtx2[j][2] and mtx1[i][2] == mtx2[j][1]:
                    res.append(
        
            
