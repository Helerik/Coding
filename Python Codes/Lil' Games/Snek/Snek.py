
import turtle
import numpy as np
import time

# Create a screen
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Snek Game")
screen.setup(1000,1000)

turtle.screensize(2000,2000)

turtle.setundobuffer(1) #limits the memory to undo.
turtle.tracer(0) #update speed

# create the map border
outline = turtle.Turtle()
outline.penup()
outline.ht()
outline.color("green")
outline.width(2)
outline.goto(-376+2, 376-2)
outline.pendown()
for _ in range(4):
    outline.fd(23.5*32)
    outline.rt(90)
outline.penup()

class snek():

    def __init__(self, init_x, init_y):

        self.head = turtle.Turtle()
        self.head.color('red')
        self.head.shape('square')
        self.head.penup()
        self.head.goto(init_x,init_y)

        self.body = [self]

        self.status = 'alive'

        self.direction = 'right'

        self.last_pos = None

        self.move_status = None

    def set_head_up(self):
        if self.direction != 'up' and self.direction != 'down':
            self.direction = 'up'

    def set_head_down(self):
        if self.direction != 'down' and self.direction != 'up':
            self.direction = 'down'

    def set_head_right(self):
        if self.direction != 'right' and self.direction != 'left':
            self.direction = 'right'

    def set_head_left(self):
        if self.direction != 'left' and self.direction != 'right':
            self.direction = 'left'

    def move(self):

        if self.move_status == 'moved':
            return

        self.last_pos = self.head.position()
        
        if self.direction == 'up':
            self.head.sety(self.head.ycor()+21)
        if self.direction == 'down':
            self.head.sety(self.head.ycor()-21)
        if self.direction == 'right':
            self.head.setx(self.head.xcor()+21)
        if self.direction == 'left':
            self.head.setx(self.head.xcor()-21)

        self.move_status = 'moved'

        if not (376-2 >= self.head.ycor() >= -(376-2)):
            self.status = 'dead'
        if not (-376+2 <= self.head.xcor() <= - (-376+2)):
            self.status = 'dead'

    def move_body(self):
        for i in range(1, len(self.body)):
            if self.collision(self.body[i]):
                self.status = 'dead'
                return
            self.body[i].move()
        self.move_status = None

    def collision(self, other):
        if self.head.position() == other.bod.position():
            return True
                
    def eat(self, other):
        if self.status == 'dead':
            return
        if self.head.position() == other.bod.position():
            self.body.append(body(self.last_pos[0], self.last_pos[1], self.body[-1]))
            other.move()


class body():

    def __init__(self, init_x, init_y, nex):

        self.bod = turtle.Turtle()
        self.bod.color('white')
        self.bod.shape('square')
        self.bod.penup()
        self.bod.goto(init_x, init_y)

        self.nex = nex

        self.last_pos = None

    def move(self):
        self.last_pos = self.bod.position()
        self.bod.goto(self.nex.last_pos[0], self.nex.last_pos[1])
        

class food():

    def __init__(self, x, y):

        self.bod = turtle.Turtle()
        self.bod.penup()
        self.bod.shape('square')
        self.bod.color('green')
        self.bod.goto(x,y)

    def move(self):
        self.bod.goto(
            21*np.random.randint(-14,14),
            21*np.random.randint(-14,14)
            )

S = snek(0, 0)

turtle.onkey(S.set_head_up, "Up")
turtle.onkey(S.set_head_down, "Down")
turtle.onkey(S.set_head_right, "Right")
turtle.onkey(S.set_head_left, "Left")
turtle.listen()

f = food(21*6, 0)

while True:
    screen.update()
    
    S.move()
    S.move_body()
    S.eat(f)

    if S.status == 'dead':
        print('Game Over')
        break

    time.sleep(0.1)







