import time
import numpy as np
import turtle

screen = turtle.Screen()
screen.bgcolor("black")
screen.title("")
screen.tracer(0)

turtle.setup(750,750)
turtle.ht()
turtle.setundobuffer(1)

class Vector():
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

class RigidBody():
    def __init__(self, x = 0, y = 0, shape = "box"):
        self.shapes = ["box"]
        
        self.position = Vector(x, y)
        self.linearVelocity = Vector()
        self.angle = 0
        self.angularVelocity = 0
        self.force = Vector()
        self.torque = 0

        self.shape = shape
        if not (isinstance(self.shape, str)) or not (self.shape in self.shapes):
            self.shape = "box"
        self.shape = self.shape.lower()
        
        self.width = 1
        self.height = 1
        self.mass = 1
        self.momentOfInertia = 0

    def ComputeInertia(self):
        if self.shape == "box":
        
            m = self.mass
            h = self.height
            w = self.width

            self.momentOfInertia = m * ((w * w) + (h * h)) / 12

    def ComputeForceTorque(self):
        f = Vector(0, 100)
        self.force = f

        r = Vector(self.width / 2, self.height / 2)
        self.torque = (r.x * f.y) - (r.y * f.x)

    def ComputeVelocity(self):
        acceleration = Vector(self.force.x / self.mass, self.force.y / self.mass)
        self.linearVelocity.x += acceleration.x * dt
        self.linearVelocity.y += (acceleration.y) * dt 

    def ComputeAngularVelocity(self):
        angular_accel = self.torque / self.momentOfInertia
        self.angularVelocity += angular_accel * dt
        self.angle += self.angularVelocity * dt

    def ComputePosition(self):
        self.position.x += self.linearVelocity.x *dt
        self.position.y += self.linearVelocity.y *dt

    def PrintRigidBody(self):
        print("body: %s; position: (%0.2f, %0.2f); angle: %0.2f" %(self.shape, self.position.x, self.position.y, self.angle))
            
startx = 0
starty = -275

box1 = RigidBody(startx, starty, "box")

box1.mass = 10
box1.angle = 0
box1.linearVelocity = Vector(0, 0)
box1.angularVelocity = 0
box1.width = 10
box1.height = 5

box1.ComputeInertia()

BOX1 = turtle.Turtle()
BOX1.shape("square")
BOX1.color("white")
BOX1.shapesize(float(box1.height), float(box1.width))
BOX1.penup()
BOX1.goto(float(startx), float(starty))
BOX1.speed(0)

def main():
    total_time = 20
    current_time = 0
    global dt
    dt = 0.01

    turtle.update()
    
    while current_time < total_time:
        time.sleep(dt)

        if current_time <= 2:
            box1.ComputeForceTorque()
        else:
            box1.torque = 0
        box1.ComputeVelocity()
        box1.ComputeAngularVelocity()
        box1.ComputePosition()

        BOX1.goto(float(box1.position.x), float(box1.position.y))
        BOX1.tiltangle(float(box1.angle))
        
        turtle.update()

        current_time += dt

        
main()        











