# Maze Game editor code

import turtle
from math import sqrt, acos, pi
import random as r
import time as t

# Create a screen
screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Maze Editor")
screen.setup(1000,1000)

turtle.screensize(2000,2000)

turtle.setundobuffer(1) #limits the memory to undo.
turtle.tracer(0) #update speed

# create the map border
outline = turtle.Turtle()
outline.penup()
outline.ht()
outline.color("grey")
outline.width(3)
outline.goto(-340,340)
outline.pendown()
for _ in range(4):
    outline.fd(340*2)
    outline.rt(90)
outline.penup()

# create pensize options border:
outline.goto(370, 35)
outline.pendown()
for _ in range(2):
    outline.fd(30*2 - 1)
    outline.rt(90)
    outline.fd(60*2)
    outline.rt(90)
outline.penup()

# Create an Editor class, which is basically a pen for editing:
class Editor(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.ht()
        self.penup()
        self.speed(0)
        self.goto(10000,10000)

        self.pen_size = 1

        self.tmp_wall = []

    # the editor pen goes to wherever the mouse clicks
    def click(self, x, y):
        self.goto(x, y)
        for button in buttons:
            button.activate()
        for wall in walls:
            wall.activate()
            
    def movearound(self, event):
        self.goto(event.x+c.canvasx(0),-1*c.canvasy(0)-event.y)
        for wall in walls:
            if wall in self.tmp_wall:
                pass
            else:
                wall.activate()

    def release(self, event):
        self.tmp_wall = []

    def increase_pensize(self):
        if self.pen_size == 2:
            pass
        else:
            self.pen_size *= 2
        
    def decrease_pensize(self):
        if self.pensize == 1:
            pass
        else:
            self.pen_size //= 2

# button class to change editor settings and commands
class Button(turtle.Turtle):
    global buttons
    buttons = []
    def __init__(self, color = "white", x = 0, y = 0, xdim = 1, ydim = 1):
        # fixed parameters
        turtle.Turtle.__init__(self)
        self.penup()
        self.speed(0)
        self.shape("square")
        # input parameters
        self.color(color)
        self.hasColor = color
        self.x = x
        self.y = y
        self.shapesize(stretch_wid = xdim, stretch_len = ydim, outline = 2)

        # function parameters, based on inputs
        self.goto(x, y)

        self.status = "inactive"

        buttons.append(self)

    def isCollision(self, other):
        if self.xcor() - self.shapesize()[0]*12 < other.xcor() < self.xcor() + self.shapesize()[0]*12 \
           and self.ycor() - self.shapesize()[1]*12 < other.ycor() < self.ycor() + self.shapesize()[1]*12:
            return True
        else:
            return False
    def activate(self):
        if self.isCollision(editor):
            self.status = "active"
            self.color("lime",self.hasColor)
            editor.goto(10000, 10000)

# Wall class which will be the building blocks of the maze's structure
class Wall(turtle.Turtle):
    def __init__(self, color, x, y):
        # fixed parameters
        turtle.Turtle.__init__(self)
        self.penup()
        self.speed(0)
        self.shape("square")

        # input parameters
        self.color(color)
        self.x = x
        self.y = y

        # function parameters, based on inputs
        self.goto(x, y)

        # status which will decide if there is a wall or not on a certain area
        self.status = "active"

    def isCollision(self, other):
        if self.xcor() - self.shapesize()[0]*12*editor.pen_size < other.xcor() < self.xcor() + self.shapesize()[0]*12*editor.pen_size \
           and self.ycor() - self.shapesize()[1]*12*editor.pen_size < other.ycor() < self.ycor() + self.shapesize()[1]*12*editor.pen_size:
            return True
        else:
            return False

    # define the status mehod
    def activate(self):
        if self.isCollision(editor):
            for i in range(editor.pen_size):
                if self.status == "active":
                    if editor.tmp_wall == []:
                        self.ht()
                        self.status = "inactive"
                        editor.tmp_wall.append(self)
                    elif editor.tmp_wall[0].status != "inactive":
                        pass
                    else:
                        self.ht()
                        self.status = "inactive"
                        editor.tmp_wall.append(self)
                elif self.status == "inactive":
                    if editor.tmp_wall == []:
                        self.st()
                        self.status = "active"
                        editor.tmp_wall.append(self)
                    elif editor.tmp_wall[0].status != "active":
                        pass
                    else:
                        self.st()
                        self.status = "active"
                        editor.tmp_wall.append(self)
            editor.goto(10000,10000)
        
walls = []
def setup_maze_initial(color = "white"):
    maze = [["X"*28]*28]
    maze = maze[0]
    
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            
            character = maze[y][x]
            screen_x = -324 + (x*24)
            screen_y =  324 - (y*24)

            if character == "X":
                walls.append(Wall(color, screen_x, screen_y))

# reads the maze created and turns it into a new list variable
def read_maze():
    global final_maze
    maze = [[]]
    for i in range(len(walls)):
        if i % 28 == 0:
            maze[-1] = "".join(maze[-1])
            maze.append([])
        if walls[i].status == "active":
            maze[-1].append("X")
        elif walls[i].status == "inactive":
            maze[-1].append(" ")
    maze[-1] = "".join(maze[-1])
    maze.remove(maze[0])
    final_maze[:] = maze[:]

loop_end = False
def end_loop():
    global loop_end
    loop_end = True

# definition of the wall color as well as the initial maze setup, for visualization purposes
try:
    editor_wallcolor = screen.textinput("Wall Color", "Choose the maze wall color")
    if editor_wallcolor == "" or editor_wallcolor == " ":
        editor_wallcolor = "white"
    setup_maze_initial(editor_wallcolor)
except:
    setup_maze_initial()

editor = Editor()

# keybinding
final_maze = []
turtle.onkey(read_maze, "s")
turtle.onkey(end_loop, "BackSpace")
turtle.listen()

def zoom(event):
    if (event.delta > 0):
        c.scale("all", 0, 0, 1.05, 1.05)
        for wall in walls:
            wall.turtlesize(wall.shapesize()[0]*1.05)
            wall.goto(wall.xcor()*1.05, wall.ycor()*1.05)
        for button in buttons:
            button.turtlesize(button.shapesize()[0]*1.05)
            button.goto(button.xcor()*1.05, button.ycor()*1.05)
    elif (event.delta < 0):
        c.scale("all", 0, 0, 1/1.05, 1/1.05)
        for wall in walls:
            wall.turtlesize(wall.shapesize()[1]*(1/1.05))
            wall.goto(wall.xcor()*(1/1.05), wall.ycor()*(1/1.05))
        for button in buttons:
            button.turtlesize(button.shapesize()[0]*(1/1.05))
            button.goto(button.xcor()*(1/1.05), button.ycor()*(1/1.05))

# mouse bindig
screen.onclick(editor.click)
c = turtle.getcanvas()
c.bind("<MouseWheel>", zoom)
c.bind("<B1-Motion>", editor.movearound)
c.bind("<ButtonRelease-1>", editor.release)
screen.listen()

# create buttons:
penup_editor = Button(x = 400, xdim = 2, ydim = 2)
pendown_editor = Button(x = 400, y = -50)

pendown_editor.color("lime", pendown_editor.hasColor)
# Editor loop:
while True:
    screen.update()
    
    if loop_end == True:
        break

    if penup_editor.status == "active":
        if editor.pen_size == 1:
            pendown_editor.status = "inactive"
            penup_editor.status = "inactive"
            pendown_editor.color(pendown_editor.hasColor)
            editor.increase_pensize()
        else:
            pass
    if pendown_editor.status == "active":
        if editor.pen_size == 2:
            penup_editor.status = "inactive"
            pendown_editor.status = "inactive"
            penup_editor.color(penup_editor.hasColor)
            editor.decrease_pensize()
        else:
            pass

if final_maze == []:
    final_maze = [["X"*28]*28]
    final_maze = final_maze[0]

print(final_maze)
turtle.bye()


