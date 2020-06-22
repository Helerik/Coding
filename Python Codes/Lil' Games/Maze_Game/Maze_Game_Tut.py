# Maze Game python code

import turtle
from math import sqrt, acos, pi
import random as r


# Create a screen
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Maze Game")
screen.setup(800,800)

turtle.setundobuffer(1) #limits the memory to undo.
turtle.tracer(0) #update speed

# Create a pen
class Pen(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.color("white")
        self.penup()
        self.speed(0)

# Create objective
class Objective(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("circle")
        self.color("yellow")
        self.penup()
        self.speed(0)

    def isNear(self, other):
        if sqrt( (self.xcor() - other.xcor())**2 + (self.ycor() - other.ycor())**2 ) < 132:
            self.st()
        else:
            self.ht()

    def isTouch(self,other):
        if self.xcor() == other.xcor() and self.ycor() == other.ycor():
            return True
        else:
            return False

# Create wall
class Wall(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.color("white")
        self.penup()
        self.speed(0)
        self.ht()

    def isNear(self, other):
        if sqrt( (self.xcor() - other.xcor())**2 + (self.ycor() - other.ycor())**2 ) < 132:
            self.st()
        else:
            self.ht()

# Create a player
class Player(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("square")
        self.color("blue")
        self.penup()
        self.speed(0)
        self.setheading(90)

    # Moving controls
    def move_up(self):
        if (self.xcor(), self.ycor() + 24) not in walls:
            self.goto(self.xcor(),self.ycor() + 24)
        
    def move_down(self):
        if (self.xcor(),self.ycor() - 24) not in walls:
            self.goto(self.xcor(),self.ycor() - 24)
            
    def move_right(self):
        if (self.xcor() + 24 ,self.ycor()) not in walls:
            self.goto(self.xcor() + 24 ,self.ycor())
            
    def move_left(self):
        if (self.xcor() - 24 ,self.ycor()) not in walls:
            self.goto(self.xcor() - 24 ,self.ycor())

# Create enemy
class Enemy(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.penup()
        self.shape("circle")
        self.color("red")
        self.speed(0)
        self.direction = r.choice(["up", "down", "left", "right"])
        self.vision = False
    
    def move(self):
        if self.vision == False:
            if self.direction == "up":
                x = 0
                y = 24
            elif self.direction == "down":
                x = 0
                y = -24
            elif self.direction == "left":
                x = -24
                y = 0
            elif self.direction == "right":
                x = 24
                y = 0
            else:
                x = 0
                y = 0

        # This defines enemy behaviour: they walk randomly until the player is both in line of sight and reachable (there are no walls in between the enemy
        # and the player). At this moment the enemy will take an optimal route to reach the player, based on the angle between itself and the player.
        # There could be an improvement to this algorithm, if the ray direction parameter was the direction determining factor of each enemy.
        elif self.vision == True:
            if self.xcor() == player.xcor() and self.ycor() == player.ycor():
                pass
            elif player.ycor() < self.ycor():
                angle = -1*((acos((player.xcor() - self.xcor())/sqrt((self.xcor() - player.xcor())**2 + (self.ycor() - player.ycor())**2)))*180/pi) + 360
                if 180 < angle < 225:
                    self.direction = "left"
                elif angle == 225:
                    self.direction = r.choice(["left", "down"])
                elif 225 < angle < 315:
                    self.direction = "down"
                elif angle == 315:
                    self.direction = r.choice(["right", "down"])
                elif 315 < angle <= 360:
                    self.direction = "right"
            else:
                angle = (acos((player.xcor() - self.xcor())/sqrt((self.xcor() - player.xcor())**2 + (self.ycor() - player.ycor())**2)))*180/pi
                if 0 <= angle < 45:
                    self.direction = "right"
                elif angle == 45:
                    self.direction = r.choice(["right", "up"])
                elif 45 < angle < 135:
                    self.direction = "up"
                elif angle == 135:
                    self.direction = r.choice(["left", "up"])
                elif 135 < angle <= 180:
                    self.direction = "left"
            self.state = "moving"
                
            if self.direction == "up":
                x = 0
                y = 24
            elif self.direction == "down":
                x = 0
                y = -24
            elif self.direction == "left":
                x = -24
                y = 0
            elif self.direction == "right":
                x = 24
                y = 0
            else:
                x = 0
                y = 0

        if (self.xcor() + x, self.ycor() + y) not in walls:
            self.goto(self.xcor() + x, self.ycor() + y)
        else:
            self.direction = r.choice(["up", "down", "left", "right"])

        turtle.ontimer(self.move, t = r.randint(300,500))

    def isNear(self, other):
        if sqrt( (self.xcor() - other.xcor())**2 + (self.ycor() - other.ycor())**2 ) < 132:
            self.st()
        else:
            self.ht()

class Ray(turtle.Turtle):
    def __init__(self, enemy):
        turtle.Turtle.__init__(self)
        self.penup()
        self.ht()
        self.shapesize(0.3,0.3)
        self.color("yellow")
        self.state = "stopped"
        self.enemy = enemy

    def isCollision(self, other):
        if sqrt((self.xcor() - other.xcor())**2 + (self.ycor() - other.ycor())**2) <= 12:
            return True
        else:
            return False

    def shoot(self, x, y):
        if self.state == "stopped":
            self.goto(x,y)
            if self.xcor() == player.xcor() and self.ycor() == player.ycor():
                pass
            elif player.ycor() < self.ycor():
                self.setheading(-1*(acos((player.xcor() - self.xcor())/sqrt((self.xcor() - player.xcor())**2 + (self.ycor() - player.ycor())**2)))*180/pi)
            else:
                self.setheading((acos((player.xcor() - self.xcor())/sqrt((self.xcor() - player.xcor())**2 + (self.ycor() - player.ycor())**2)))*180/pi)
            self.state = "moving"
    
    def move(self):
        if self.state == "moving":
            for _ in range(16):
                self.fd(6)
                for wall in wall_obj:
                    if self.isCollision(wall):
                        self.state = "stopped"
                        self.enemy.vision = False
                        break
                if self.state == "stopped":
                    break
                if self.isCollision(player):
                    self.state = "stopped"
                    self.enemy.vision = True
                    break
                else:
                    self.enemy.vision = False
            self.state = "stopped"
                
 
            
        
    

mazes = []

lvl_1 = [
"XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
"XPXXX           XXXX X XXXXX",
"X XXXX  XXXXXXX XXXX X XXXXX",
"X       X       XXXX X    OX",
"X       XXXXXXXXXXXX X  XXXX",
"X    E  X            X  XXXX",
"X       X  E  XXXXXXXX  XXXX",
"X       X                  X",
"X       XXXXXXXXXXXXXXXX XXX",
"X                     X    X",
"X     XXXXXXXXXXXXXXX XXX  X",
"X                XXXX   X  X",
"X     XXXXXXXXXXXXXXXXXXXX X",
"X                      X   X",
"X     E    XXXXXXXXXXX X   X",
"X          X     X   X X   X",
"XXXXXXXXXXXX  E  X   X XXX X",
"X          X     X       X X",
"X  XXXXXXXXXXX   X  XXXX X X",
"X     X      X   X  X  X X X",
"X  X  X    X X XXX  X  X X X",
"X  X  X    X X XXX  X  X X X",
"X  X  X    X   XXX  X  X X X",
"X  XXXX    XXXXXXX  X XX X X",
"X  X                X    X X",
"X  XXXX    XXXXXXXXXXXXXXX X",
"X                    E     X",
"XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
]

mazes.append(lvl_1)

def setup_maze(lvl):
    for y in range(len(lvl)):
        for x in range(len(lvl[y])):
            character = lvl[y][x]
            screen_x = -324 + (x*24)
            screen_y =  324 - (y*24)

            if character == "X":
                wall_tmp = Wall()
                wall_tmp.goto(screen_x, screen_y)
                wall_obj.append(wall_tmp)
                walls.append((screen_x, screen_y))

            if character == "P":
                player.goto(screen_x, screen_y)

            if character == "O":
                objective.goto(screen_x, screen_y)

            if character == "E":
                enemies.append(Enemy())
                enemies[len(enemies)-1].goto(screen_x, screen_y)
                rays.append(Ray(enemies[len(enemies)-1]))
                turtle.ontimer(enemies[len(enemies)-1].move, t = 1)
                
                

# Define Walls:
walls = []
wall_obj = []

# Define objective:
objective = Objective()

# Define enemies
enemies = []

# Define rays
rays = []

# Define player:
player = Player()
# Keyinding:
turtle.onkey(player.move_up, "Up")
turtle.onkey(player.move_down, "Down")
turtle.onkey(player.move_right, "Right")
turtle.onkey(player.move_left, "Left")
turtle.listen()

setup_maze(lvl_1)

# Main game loop:
while True:
    screen.update()

    for ray in rays:
        ray.shoot(ray.enemy.xcor(),ray.enemy.ycor())
        ray.move()

    for wall in wall_obj:
        wall.isNear(player)
    objective.isNear(player)

    for enemy in enemies: 
        enemy.isNear(player)

    if objective.isTouch(player):
        screen.update()
        break
    






