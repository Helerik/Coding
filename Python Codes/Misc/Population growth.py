
import numpy as np
import time
import matplotlib.pyplot as plt

class herbivore():
    
    def __init__(self, size, speed, init_age, max_age,
                 energy_need, reproduction_age, birth_probability,
                 mutation_prob, birth_penalty, speed_penalty):
        
        self.size = size
        self.speed = speed
        self.age = init_age
        self.max_age = max_age
        
        self.energy = 500*self.size
        self.max_energy = 500*self.size
        self.energy_need = energy_need
        
        self.reproduction_age = reproduction_age
        self.birth_prob = birth_probability

        self.birth_penalty = birth_penalty
        self.speed_penalty = speed_penalty

        self.status = 'alive'       

        self.mutation_prob = mutation_prob

    def age_up(self):

        self.age += 1
        
        if self.age >= self.max_age:
            self.die()
            return

        if (self.reproduction_age + self.max_age)*np.power(self.size, 3)/3 >= self.age >= self.reproduction_age and self.energy > self.energy_need:
            
            if np.random.random() < self.birth_prob:
                
                n_age = 0
                
                if np.random.random() < self.mutation_prob:
                    n_size = np.random.normal(self.size, 0.5)
                    if n_size < 0.05:
                        n_size = 0.05
                else:
                    n_size = self.size
                    
                if np.random.random() < self.mutation_prob:
                    n_speed = np.random.normal(self.speed, 0.5)
                    if n_speed < 0:
                        n_speed = 0
                else:
                    n_speed = self.speed
                    
                if np.random.random() < self.mutation_prob:
                    n_max_age = np.random.normal(self.max_age, 0.5)
                    if n_max_age < 0:
                        n_max_age = 0
                else:
                    n_max_age = self.max_age
                    
                if np.random.random() < self.mutation_prob:
                    n_energy_need = np.random.normal(self.energy_need, 0.5)
                    if n_energy_need < 50:
                        n_energy_need = 50
                else:
                    n_energy_need = self.energy_need
                    
                if np.random.random() < self.mutation_prob:
                    n_reproduction_age = np.random.normal(self.reproduction_age, 0.5)
                    if n_reproduction_age >= n_max_age:
                        n_reproduction_age = n_max_age
                    if n_reproduction_age < 5:
                        n_reproduction_age = 5
                else:
                    n_reproduction_age = self.reproduction_age
                    
                if np.random.random() < self.mutation_prob:
                    n_birth_prob = np.random.normal(self.birth_prob, 0.1)
                    if n_birth_prob >= 1:
                        n_birth_prob = 1
                    if n_birth_prob < 0:
                        n_birth_prob = 0
                else:
                    n_birth_prob = self.birth_prob
                    
                if np.random.random() < self.mutation_prob:
                    n_mutation_prob = np.random.normal(self.mutation_prob, 0.1)
                    if n_mutation_prob >= 1:
                        n_mutation_prob = 1
                    if n_mutation_prob < 0:
                        n_mutation_prob = 0
                else:
                    n_mutation_prob = self.mutation_prob
                    
                if np.random.random() < self.mutation_prob:
                    n_birth_penalty = np.random.normal(self.birth_penalty, 0.1)
                    if n_birth_penalty < 0:
                        n_birth_penalty = 0
                else:
                    n_birth_penalty = self.birth_penalty
                if np.random.random() < self.mutation_prob:
                    n_speed_penalty = np.random.normal(self.speed_penalty, 0.1)
                    if n_speed_penalty < 0:
                        n_speed_penalty = 0
                else:
                    n_speed_penalty = self.speed_penalty

                herbivores.append(herbivore(n_size, n_speed, n_age, n_max_age, n_energy_need, n_reproduction_age, n_birth_prob, n_mutation_prob,
                                            n_birth_penalty, n_speed_penalty))
                self.energy -= self.birth_penalty

    def die(self):
        self.status = 'dead'
        return

    def eat(self, food_count, nutrition_val):
        
        if np.random.random() < ((food_count/len(herbivores)) + (1 - 1/(self.speed + 1)))/2:
            self.energy += nutrition_val/np.power(self.size, 3)
            if self.energy > 500*self.size:
                self.energy = 500*self.size
            food_count -= 1
        elif self.speed <= 1:
            self.energy -= self.speed_penalty * np.power(self.size, 3)
        else:
            self.energy -= self.speed_penalty * np.square(self.speed) * np.power(self.size, 3)

        if food_count < 0:
            food_count = 0
            
        return food_count

    def spend_energy(self):
        if self.energy < 0:
            self.energy = 0
        if self.energy < self.energy_need:
            self.max_energy -= 100
        else:
            self.max_energy += 50

        if self.max_energy < 100:
            self.die()

global herbivores
herbivores = []
food_count = 1000
nutrition_val = 250

n = 100
for i in range(n):
    he_init = herbivore(size = 3, speed = 3, init_age = int(np.random.normal(20,2)), max_age = 60, energy_need = 100, reproduction_age = 30,
                        birth_probability = 0.35, mutation_prob = 0.1, birth_penalty = 1000, speed_penalty = 400)
    herbivores.append(he_init)
k = -1
x = []
y = []
y1 = []
y_means = []

sim_speed = 1000

plt.figure('Population growth simulation')
while True:
    j = 0
    k += 1
    for i in range(len(herbivores)):
        food_count = herbivores[i+j].eat(food_count, nutrition_val)
        herbivores[i+j].age_up()
        herbivores[i+j].spend_energy()

        if herbivores[i+j].status != 'alive':
            herbivores.pop(i+j)
            j -= 1

    
    food_count += np.random.normal(200,2)

    x.append(k)
    y.append(len(herbivores))
##    y1.append(food_count)
    y_means.append(np.mean(y))

    if k >= 500:
        x.pop(0)
        y.pop(0)
##        y1.pop(0)
        y_means.pop(0)

    plt.clf()
    plt.xlabel('Time elapsed')
    plt.ylabel('Population size')
    plt.title('Population Growth')
    plt.plot(x,y, color = 'r')
    plt.axhline(np.mean(y), color = 'b', lw = 1)
    plt.plot(x,y_means, color = 'y', lw = 1)
##    plt.plot(x,y1, color = 'g')
    plt.pause(1/sim_speed)

    if k%10 == 0:
        print('Iter %d food amount = %d' %(k, food_count), '\n')

    av_size = 0
    av_speed = 0
    av_age = 0
    av_energy_need = 0
    av_rep_age = 0
    av_birth_prob = 0
    av_birth_penalty = 0
    av_speed_penalty = 0
    if k%500 == 0:
        for he in herbivores:
            av_size += he.size
            av_speed += he.speed
            av_age += he.max_age
            av_energy_need += he.energy_need
            av_rep_age += he.reproduction_age
            av_birth_penalty += he.birth_penalty
            av_speed_penalty += he.speed_penalty
            av_birth_prob += he.birth_prob
        m = len(herbivores)
        av_size /= m
        av_speed /= m
        av_age /= m
        av_energy_need /= m
        av_rep_age /= m
        av_birth_penalty /= m
        av_speed_penalty /= m
        av_birth_prob /= m

        print('''%s \nAverages:\nSpeed = %f \nSize = %f \nMax age = %f \nEnergy need = %f
Reproduction age = %f \nBirth penalty = %f \nSpeed penalty = %f \nBirth probability = %f \n%s\n'''
              %(40*'=', av_speed, av_size, av_age, av_energy_need, av_rep_age, av_birth_penalty, av_speed_penalty, av_birth_prob,40*'='))

        time.sleep(5)
            

    if len(herbivores) == 0 or len(herbivores) >= 50000:
        plt.clf()
        plt.plot(x,y, color = 'b')
        plt.show()
        break
    
##
##
##
##
##
##
##
##
##Iter 13440 food amount = 206 
##
##Iter 13450 food amount = 202 
##
##Iter 13460 food amount = 202 
##
##Iter 13470 food amount = 195 
##
##Iter 13480 food amount = 197 
##
##Iter 13490 food amount = 202 
##
##Iter 13500 food amount = 202 
##
##======================================== 
##Averages:
##Speed = 6.063659 
##Size = 1.333544 
##Max age = 60.210420 
##Energy need = 99.617504
##Reproduction age = 29.688641 
##Birth penalty = 1000.016681 
##Speed penalty = 399.883654 
##Birth probability = 0.560899 
##========================================
##
##Iter 13510 food amount = 201 
##
##Iter 13520 food amount = 200 
##
##Iter 13530 food amount = 200 
##
##Iter 13540 food amount = 201 
##
##Iter 13550 food amount = 200 
##
##Iter 13560 food amount = 198 
##
##Iter 13570 food amount = 203 
##
##



        










            
           
