# Space Wars (Old game recreation)
# USING TURTLE MODULE

import os
import winsound
import random as r
import turtle
import time as t

turtle.fd(0)
turtle.speed(0)
turtle.bgcolor("black")
turtle.bgpic("space.gif")
turtle.title("Space Wars")
turtle.ht() #hideturtle()
turtle.setundobuffer(1) #limits the memory to undo.
turtle.tracer(0)

# Create game and screen:
class Game:
    def __init__(self):
        self.level = 1
        self.score = 0
        self.state = "playing"
        self.pen = turtle.Turtle()

    def draw_border(self):
        self.pen.speed(0)
        self.pen.color("white")
        self.pen.pensize(3)
        self.pen.penup()
        self.pen.goto(-300,300)
        self.pen.pendown()
        for _ in range(4):
            self.pen.fd(600)
            self.pen.rt(90)
        self.pen.penup()
        self.pen.ht()
        self.pen.pendown()

    def show_status(self):
        msg = "Score: %.10d" %(self.score) + " "*10 + "Speed: %10s" %(player.speed*"|") + " "*10 + "Lives: %.3d" %(player.lives)
        self.pen.undo()
        self.pen.penup()
        try:
            self.pen.goto(-300,310)
            self.pen.write(msg, font = ("Arial", 16, "normal"))
        except:
            try:
                self.pen.goto(-300,310)
                self.pen.write(msg, font = ("Arial", 16, "normal"))
            except:
                self.pen.goto(-300,310)
                self.pen.write(msg, font = ("Arial", 16, "normal")) 

game = Game()
game.draw_border()        

# The objects of the game:
class Sprite(turtle.Turtle): # inherits everything from the turtle module
    def __init__(self, spriteshape, color, startx, starty):
        turtle.Turtle.__init__(self, shape = spriteshape)
        self.speed(0)
        self.penup()
        self.color(color)
        self.fd(0)
        self.goto(startx, starty)
        self.speed = 1

    # All sprites have this method.
    def move(self):
        self.fd(self.speed)

        # define boundary detection

        if self.xcor() > 290:
            self.setx(290)
            if self.heading() < 270 and self.heading() != 0:
                self.lt(90)
            elif self.heading() > 270 and self.heading() != 0:
                self.rt(90)
            else:
                self.lt(180)
                
        if self.xcor() < -290:
            self.setx(-290)
            if self.heading() > 180:
                self.lt(90)
            elif self.heading() < 180:
                self.rt(90)
            else:
                self.lt(180)
                
        if self.ycor() > 290:
            self.sety(290)
            if self.heading() > 90:
                self.lt(90)
            elif self.heading() < 90:
                self.rt(90)
            else:
                self.lt(180)
                
        if self.ycor() < -290:
            self.sety(-290)
            if self.heading() > 270:
                self.lt(90)
            elif self.heading() < 270:
                self.rt(90)
            else:
                self.lt(180)

    # This is a square box collision.
    def isCollision(self, other):
        if self.xcor() >= other.xcor() - 15 and self.xcor() <= other.xcor() + 15 and self.ycor() >= other.ycor() - 15 and self.ycor() <= other.ycor() + 15:
            return True
        else:
            return False

class Player(Sprite):

    def __init__(self, spriteshape, color, startx, starty):
        Sprite.__init__(self, spriteshape, color, startx, starty)
        self.shapesize(0.8,1)
        self.speed = 3
        self.lives = 3

    def turn_left(self):
        self.lt(45)

    def turn_right(self):
        self.rt(45)

    def accelerate(self):
        turtle.onkey(None, "Down")
        turtle.listen()
        if self.speed == 10:
            pass
        else:
            winsound.PlaySound("1.wav", winsound.SND_ASYNC)
            self.speed += 1
            try:
                game.show_status()
            except:
                try:
                    game.show_status()
                except:
                    game.show_status()
        turtle.onkey(player.decelerate, "Down")
        turtle.listen()

    def decelerate(self):
        turtle.onkey(None, "Up")
        turtle.listen()
        if self.speed == 0:
            pass
        else:
            winsound.PlaySound("1.wav", winsound.SND_ASYNC)
            self.speed -= 1
            try:
                game.show_status()
            except:
                try:
                    game.show_status()
                except:
                    game.show_status()
        turtle.onkey(player.accelerate, "Up")
        turtle.listen()

class Missile(Sprite):

    def __init__(self, spriteshape, color, startx, starty):
        Sprite.__init__(self, spriteshape, color, startx, starty)
        self.shapesize(0.25, 0.7)
        self.speed = 20
        self.status = "ready"
        self.goto(-1000,1000)

    def fire(self):
        if self.status == "ready":
            self.goto(player.xcor(),player.ycor())
            self.setheading(player.heading())
            self.status = "firing"
            winsound.PlaySound("4.wav", winsound.SND_ASYNC)

    def move(self):
        if self.status == "firing":
            self.fd(self.speed)

        # Border check:
            if self.xcor() < -290 or self.xcor() > 290 \
            or self.ycor() < -290 or self.ycor() > 290:
                self.goto(-1000,1000)
                self.status = "ready"

        else:
            self.goto(-1000,1000)

class Ally(Sprite):
    def __init__(self, spriteshape, color, startx, starty):
        Sprite.__init__(self, spriteshape, color, startx, starty)

        self.shapesize(0.9,1)

        self.speed = 8
        self.setheading(r.randint(0,360))

class Enemy(Sprite):

    def __init__(self, spriteshape, color, startx, starty):
        Sprite.__init__(self, spriteshape, color, startx, starty)

        self.shapesize(0.9,1)
        
        self.speed = 6
        self.setheading(r.randint(0,360))

class Particle(Sprite):
    def __init__(self, spriteshape, color, startx, starty):
        Sprite.__init__(self, spriteshape, color, startx, starty)

        self.goto(-1000,1000)
        self.shapesize(0.1,0.1)
        self.frame = 1
        self.setheading(r.randint(0,360))        

    def move(self):
        self.fd(10)
        self.frame += 1

        if self.frame >= 20 or self.xcor() > 290 or self.xcor() < -290 or self.ycor()>290 or self.ycor() <-290:
            self.ht()
            
# Pause variable:
paused = False
pause_pen = turtle.Turtle()
pause_pen.ht()
pause_pen.penup()
pause_pen.color("white")
pause_pen.goto(-70,0)

# Pause functions:
def pause():
    global paused
    global pause_pen
    if paused == True:
        for enemy in enemies:
            enemy.st()
        for ally in allies:
            ally.st()
        missile.st()
        player.st()
        turtle.onkey(player.turn_left, "Left")
        turtle.onkey(player.turn_right, "Right")
        turtle.onkey(player.accelerate, "Up")
        turtle.onkey(player.decelerate, "Down")
        turtle.onkey(missile.fire, "space")
        turtle.listen()
        pause_pen.clear()
        paused = False
        return
    else:
        for enemy in enemies:
            enemy.ht()
        for ally in allies:
            ally.ht()
        missile.ht()
        player.ht()
        turtle.onkey(None, "Left")
        turtle.onkey(None, "Right")
        turtle.onkey(None, "Up")
        turtle.onkey(None, "Down")
        turtle.onkey(None, "space")
        turtle.listen()
        pause_pen.write("        PAUSED\n", font = ("Arial", 14, "normal"))
        pause_pen.write("- CLICK TO EXIT -", font = ("Arial", 14, "normal"))
        paused = True
        return

# Create sprites:
player = Player("triangle", "white", 0, 0)
game.show_status()
missile = Missile("triangle", "yellow", 0, 0)

enemies = []
allies = []
for i in range(6):
    enemies.append(Enemy("circle", "red", -100, 0))
    allies.append(Ally("square", "blue", 100, 0))

# Keyboard biding:

turtle.onkey(player.turn_left, "Left")
turtle.onkey(player.turn_right, "Right")
turtle.onkey(player.accelerate, "Up")
turtle.onkey(player.decelerate, "Down")
turtle.onkey(missile.fire, "space")
turtle.onkey(pause, "p")
turtle.listen()

# Turtle exit function
def bye_bye(x, y):
    turtle.bye()

# Main game loop:
while True:
    # Pause loop
    while paused == True:
        turtle.onscreenclick(bye_bye)
        turtle.update()
        t.sleep(0.05)
        
    end = -1
    turtle.update()
    t.sleep(0.05)

    player.move()
    missile.move()

    for enemy in enemies:
        enemy.move()
        
        if player.isCollision(enemy):
            turtle.onkey(None, "Up")
            turtle.onkey(None, "Down")
            turtle.listen()
            
            player.lives -= 1

            # Game Over loop:
            
            if player.lives < 0:

                turtle.onkey(None, "Left")
                turtle.onkey(None, "Right")
                turtle.onkey(None, "Up")
                turtle.onkey(None, "Down")
                turtle.onkey(None, "space")
                turtle.listen()

                turtle.tracer(1)
                x = 1
                y = 1
                t.sleep(0.5)
                for _ in range(24*4):
                    t.sleep(0.001)
                    player.lt(15)
                    player.shapesize(x,y)
                    x -= 0.0104166667
                    y -= 0.0104166667
                player.hideturtle()
                turtle.tracer(0)
                particles = []
                for i in range(50):
                    particles.append(Particle("square", "white", 0, 0))
                for particle in particles:
                    particle.goto(player.xcor(), player.ycor())
                winsound.PlaySound("5.wav", winsound.SND_ASYNC)
                for particle in range(20):
                    turtle.update()
                    t.sleep(0.05)
                    for particle in particles:
                        particle.move()
                end = "end"
                break
            
            else:
                winsound.PlaySound("2.wav", winsound.SND_ASYNC)
                
            xran = r.random()
            yran = r.random()
            if xran < 0.5:
                x = player.xcor() + r.randint(50,250)
            else:
                x = player.xcor() + r.randint(-250,-50)
            if yran < 0.5:
                y = player.ycor() + r.randint(50,250)
            else:
                y = player.ycor() + r.randint(-250,-50)
            if x > 290:
                x = 290
            if x < -290:
                x = -290
            if y > 290:
                y = 290
            if y < -290:
                y = -290
            enemy.setheading(r.randint(0,360))
            enemy.goto(x,y)
            game.score -= 100
            if game.score < 0:
                game.score = 0
            try:
                game.show_status()
            except:
                try:
                    game.show_status()
                except:
                    game.show_status()
            turtle.onkey(player.accelerate, "Up")
            turtle.onkey(player.decelerate, "Down")
            turtle.listen()
                    
        if missile.isCollision(enemy):
            turtle.onkey(None, "Up")
            turtle.onkey(None, "Down")
            turtle.listen()
            xran = r.random()
            yran = r.random()
            if xran < 0.5:
                x = player.xcor() + r.randint(25,250)
            else:
                x = player.xcor() + r.randint(-250,-25)
            if yran < 0.5:
                y = player.ycor() + r.randint(25,250)
            else:
                y = player.ycor() + r.randint(-250,-25)
            if x > 290:
                x = 290
            if x < -290:
                x = -290
            if y > 290:
                y = 290
            if y < -290:
                y = -290
            enemy.setheading(r.randint(0,360))
            enemy.goto(x,y)
            missile.status = "ready"
            missile.move()
            game.score += 100
            try:
                game.show_status()
            except:
                try:
                    game.show_status()
                except:
                    game.show_status()
            winsound.PlaySound("2.wav", winsound.SND_ASYNC)
            turtle.onkey(player.accelerate, "Up")
            turtle.onkey(player.decelerate, "Down")
            turtle.listen()

    if end == "end":
        break
            
    for ally in allies:
        ally.move()
        
        if missile.isCollision(ally):
            turtle.onkey(None, "Up")
            turtle.onkey(None, "Down")
            turtle.listen()
            
            x = r.randint(-250,250)
            y = r.randint(-250,250)
            ally.goto(x,y)
            missile.status = "ready"
            missile.move()
            game.score -= 50
            if game.score < 0:
                game.score = 0
            try:
                game.show_status()
            except:
                try:
                    game.show_status()
                except:
                    game.show_status()
            winsound.PlaySound("2.wav", winsound.SND_ASYNC)
            turtle.onkey(player.accelerate, "Up")
            turtle.onkey(player.decelerate, "Down")
            turtle.listen()
        
            
with open("scores.txt", "a") as file:
    file.write(str(game.score)+"\n")
    file.close()
with open("scores.txt", "r") as file:
    num_list = [int(num) for num in file.read().split()]

    max_val = max(num_list)

    file.close()

scores = turtle.Turtle()
scores.ht()
scores.color("white")
scores.penup()
scores.goto(-100,100)
scores.pendown()
for _ in range(4):
    scores.fd(200)
    scores.rt(90)
scores.penup()
scores.goto(-90,70)
scores.write("Highscore: " + str(max_val), font = ("Arial", 14, "normal"))
scores.goto(-90,40)
scores.write("Your score: " + str(game.score), font = ("Arial", 14, "normal"))

turtle.exitonclick()
































