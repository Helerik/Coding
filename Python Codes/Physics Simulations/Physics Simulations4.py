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

    def __str__(self):
        return ("(%f ; %f)" %(self.x, self.y))

class AABB():
    def __init__(self):
        self.min = Vector()
        self.max = Vector()

    def isOverlap(self, other):
        d1x = other.min.x - self.max.x;
        d1y = other.min.y - self.max.y
        d2x = self.min.x - other.max.x
        d2y = self.min.y - other.max.y

        if d1x > 0 or d1y > 0:
            return False

        if d2x > 0 or d2y > 0:
            return False

        return True

class RigidBodyBox():
    def __init__(self, x = 0, y = 0):
        
        self.position = Vector(x, y)
        self.linearVelocity = Vector()
        self.angle = 0
        self.angularVelocity = 0
        self.force = Vector()
        self.torque = 0
        
        self.shape = "box"
        self.width = 1
        self.height = 1
        self.mass = 1
        self.momentOfInertia = 0

    def ComputeInertia(self):
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
        self.linearVelocity.y += acceleration.y * dt 

    def ComputeAngularVelocity(self):
        angular_accel = self.torque / self.momentOfInertia
        self.angularVelocity += angular_accel * dt
        self.angle += self.angularVelocity * dt

    def ComputePosition(self):
        self.position.x += self.linearVelocity.x * dt
        self.position.y += self.linearVelocity.y * dt

    def PrintRigidBodyBox(self):
        print("body: %s; position: (%0.2f, %0.2f); angle: %0.2f" %(self.shape, self.position.x, self.position.y, self.angle))


class RigidBodyCircle():
    def __init__(self, x = 0, y = 0):
        
        self.position = Vector(x, y)
        self.linearVelocity = Vector()
        self.angle = 0
        self.angularVelocity = 0
        self.force = Vector()
        self.torque = 0
        
        self.shape = "circle"
        self.radius = 1
        self.mass = 1
        self.momentOfInertia = 0

        self.AABB = AABB()

    def ComputeInertia(self):
        m = self.mass
        r = self.radius

        self.momentOfInertia = m * r * r / 2

    def ComputeForceTorque(self):
        f = Vector(0, 100)
        self.force = f

        r = Vector(self.radius / 2, self.radius / 2)
        self.torque = (r.x * f.y) - (r.y * f.x)

    def ComputeVelocity(self):
        acceleration = Vector(self.force.x / self.mass, self.force.y / self.mass)   
        self.linearVelocity.x += acceleration.x * dt
        self.linearVelocity.y += acceleration.y * dt 

    def ComputeAngularVelocity(self):
        angular_accel = self.torque / self.momentOfInertia
        self.angularVelocity += angular_accel * dt
        self.angle += self.angularVelocity * dt

    def ComputePosition(self):
        self.position.x += self.linearVelocity.x * dt
        self.position.y += self.linearVelocity.y * dt

    def PrintRigidBodyCircle(self):
        print("body: %s; position: (%0.2f, %0.2f); angle: %0.2f" %(self.shape, self.position.x, self.position.y, self.angle))

    def isCollision(self, other):
        x = self.position.x - other.position.x
        y = self.position.y - other.position.y

        squareNorm = (x * x) + (y * y)
        radius = self.radius + other.radius
        sqrradius = radius * radius

        if squareNorm <= sqrradius:
            return True
        else:
            return False

    def reworkAABB(self):
        self.AABB.min = Vector(self.position.x - self.radius, self.position.y - self.radius)
        self.AABB.max = Vector(self.position.x + self.radius, self.position.y + self.radius)
        

circle1 = RigidBodyCircle(0, -200)
circle2 = RigidBodyCircle(0, -300)

circle1.radius = 2
circle1.mass = 2
circle1.ComputeInertia()

circle2.ComputeInertia()

CIR1 = turtle.Turtle()
CIR1.shape("circle")
CIR1.color("white")
CIR1.shapesize(2)
CIR1.penup()
CIR1.goto(0, -200)
CIR1.speed(0)

CIR2 = turtle.Turtle()
CIR2.shape("circle")
CIR2.color("white")
CIR2.shapesize(1)
CIR2.penup()
CIR2.goto(0, -300)
CIR2.speed(0)

circle1.reworkAABB()
circle2.reworkAABB()

def main():
    t = time.time()
    total_time = 20
    current_time = 0
    global dt
    dt = 0.01

    turtle.update()
    
    while current_time < total_time:
        time.sleep(dt)

        circle1.ComputeForceTorque()
        circle2.ComputeForceTorque()
        
        circle1.ComputeVelocity()
        circle1.ComputeAngularVelocity()
        circle1.ComputePosition()

        circle2.ComputeVelocity()
        circle2.ComputeAngularVelocity()
        circle2.ComputePosition()

        circle1.reworkAABB()
        circle2.reworkAABB()
        if circle1.AABB.isOverlap(circle2.AABB) or circle2.AABB.isOverlap(circle1.AABB):
            print("AABB overlap")

        CIR1.goto(float(circle1.position.x), float(circle1.position.y))
        CIR1.tiltangle(float(circle1.angle))

        CIR2.goto(float(circle2.position.x), float(circle2.position.y))
        CIR2.tiltangle(float(circle2.angle))
        
        turtle.update()

        current_time += dt
        
        
main()        











