import random
import matplotlib.pyplot as plt
import time

print("Done importing\n")

def sort(List):

    while 1:
        key = 1
        for i in range(len(List)):
            
            if i == len(List)-1:
                pass
            else:
                if List[i] <= List[i+1]:
                    pass
                else:
                    key = 0
                    List.append(List.pop(i))
        if key:
            return List

time_lis = [] 
for i in range(1, 70+1):
    avrg_time = 0
    for j in range(100):
        lis = [random.random() for _ in range(i)]
        t = time.time()
        sort(lis)
        avrg_time += time.time() - t
    avrg_time /= 100
    time_lis.append(avrg_time)

plt.plot(time_lis)
plt.show()
