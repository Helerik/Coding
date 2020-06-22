def pivot(A):

    ID = []
    for i in range(len(A)):
        ID.append([])
        for j in range(len(A[i])):
            ID[i].append(0)
            if i == j:
                ID[i][j] = 1    
    for i in range(len(A)):
        maxlin = max(range(i, len(A)), key=lambda j: abs(A[j][i]))
        if i != maxlin:
            ID[i], ID[maxlin] = ID[maxlin], ID[i]

    return ID

print(pivot([[0, 2, 3], [4, 0, 6], [7, 8, 0]]))
                
