import numpy as np
import time

class Vector():
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

class Particle():
    def __init__(self):
        self.position = Vector()
        self.velocity = Vector()
        self.mass = 1
        self.force = Vector()

    def computeForce(self):
        self.force = Vector(0, self.mass*(-9.81))

    def computeVelocity(self):
        self.velocity = Vector(round(self.velocity.x + dt * self.force.x / self.mass, 2), round(self.velocity.y + dt * self.force.y / self.mass, 2))

    def computePosition(self):
        self.position = Vector(round(self.position.x + self.velocity.x * dt, 2), round(self.position.y + self.velocity.y * dt, 2))

par = Particle()
par.position = Vector(-8,57)
par.velocity = Vector(0,0)


def runSim():
    total_time = 10
    current_time = 0
    global dt
    dt = 1 # 1 second per step

    print(par.position.x, par.position.y)
    
    while current_time < total_time:
        time.sleep(dt)

        par.computeForce()
        par.computeVelocity()
        par.computePosition()

        print(par.position.x, par.position.y)

        current_time += dt

runSim()
