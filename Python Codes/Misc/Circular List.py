# !/usr/bin/env python3
# Author: Erik Davino Vincent

class LoopList():

    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        return self._data[idx % len(self._data)]

print("Example of LoopList working\n")
l = LoopList([1,2,'a'])
for i in range(-5, 5):
    print(l[i])

print()
    


    

    
