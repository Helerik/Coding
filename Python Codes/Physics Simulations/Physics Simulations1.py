from math import *
import numpy as np
import turtle
import time

screen = turtle.Screen()
turtle.setup(750,750)

screen.bgcolor("black")
screen.title("")
screen.tracer(0)

turtle.ht() #hideturtle()
turtle.setundobuffer(1) #limits the memory to undo.

''' 10 pxls = 1 m '''
''' 1 loop = 1 sec '''

global gravity
gravity = -9.807
gravity /= 10

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

class rigid_body(turtle.Turtle):
    def __init__(self, color, shape, size, mass, friction_constant, startx = 0, starty = 0):
        turtle.Turtle.__init__(self)

        self.shape(shape)
        self.color(color)
        self.shapesize(size/20, size/20)
        self.speed(0)
        self.penup()
        self.goto(startx, starty)

        self.fric = friction_constant
        self.mass = mass
        self.dx = 0
        self.dy = 0

    def fall(self):
        self.dy += gravity
        self.sety(self.ycor() + self.dy)

    def move(self):
        self.setx(self.xcor() + self.dx)

    def bounce(self):
        
        if self.ycor() <= -300:
            self.goto(self.xcor(), -300)
            self.dy *= -0.75 + np.random.uniform(0, 0.01)

        if np.abs(self.xcor()) >= 300:
            self.goto(np.sign(self.xcor())*300, self.ycor())
            self.dx *= -0.75 + np.random.uniform(0, 0.01)

        if self.ycor() <= -300:
            if self.dx >= 0:
                self.dx -= self.fric*(-gravity)
                if self.dx <= 0.00001:
                    self.dx = 0
            if self.dx <= 0:
                self.dx += self.fric*(-gravity)
                if self.dx >= -0.00001:
                    self.dx = 0

    def iscollision(self, other):
        if np.sqrt((self.xcor()-other.xcor())**2 + (self.ycor()-other.ycor())**2) <= 20:
            return True
                   
        
pen = turtle.Turtle()
pen.color("white")
pen.penup()
pen.ht()
pen.goto(-310,-310)
pen.pendown()
for i in range(4):
    pen.fd(620)
    pen.lt(90)
pen.penup()

key = 0
square = rigid_body("white", "circle", 20, 1, 0.3, startx = 0, starty = 0)
square1 = rigid_body("white", "circle", 20, 1, 0.3, startx = 0, starty = -100)

while True:
    turtle.update()
    time.sleep(0.05)
    
    square.move()
    square.fall()
    square.bounce()

    square1.move()
    square1.fall()
    square1.bounce()

    print(square.iscollision(square1))

        
