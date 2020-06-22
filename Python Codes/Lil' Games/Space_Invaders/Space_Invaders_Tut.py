# Space Invaders tutorial
# 16/07/2019

import turtle
turtle.setundobuffer(1)
turtle.tracer(1)
import os
import math
import random as r
import time as t
import winsound as wins # caso queira adicionar som

## For windows add the following (Thanks for helping me make my first game after many failed attempts finally found someone that explains things well)
##
## Place explosion.wav and laser.wav in the same folder as your game.py file
## import winsound
##
## for the explosion add           winsound.PlaySound("explosion", winsound.SND_ASYNC)
## for the laser add           winsound.PlaySound("laser", winsound.SND_ASYNC)

end = False

# Criar tela para o jogo

screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Space Invaders")
screen.bgpic("space_background.gif") # adiciona imagem de fundo

# Podemos registrar imagens para serem nossos objetos

turtle.register_shape("invader.gif")
turtle.register_shape("player.gif")

# Desenhar borda para a tela

borda = turtle.Turtle()
borda.speed(0) # 0 eh a maior velocidade possivel
borda.color("White")
borda.penup() # faz com que ao mover a pen, nao crie um traco.
borda.setposition(-300,-300)
borda.pendown() # permite que a pen crie um traco
borda.pensize(3)
for side in range(4):
    borda.fd(600)
    borda.lt(90)
borda.hideturtle()

# Definimos a pontuacao (sistema)

score = 0

score_pen = turtle.Turtle()
score_pen.speed(0)
score_pen.color("white")
score_pen.penup()
score_pen.setposition(-290,275)
scorestring = "Score: %d" %score
score_pen.write(scorestring, False, align="left", font = ("Arial", 14, "normal"))
score_pen.hideturtle()
    
# Criar um jogador

player = turtle.Turtle()
player.hideturtle()
player.color("blue")
player.shape("player.gif")
player.penup()
player.speed(0)
player.lt(90)
player.setposition(0,-250)
player.showturtle()

# Criamos o projetil do jogador

bala = turtle.Turtle()
bala.hideturtle()
bala.color("red")
bala.shape("square")
bala.shapesize(0.1,0.5)
bala.penup()
bala.speed(0)
bala.setheading(90)

balaspd = 20

bala_state = "pronta"

# Cria inimigos
# Definimos o numero de inimigos

enemy_num = 5

enemies = []

# Adicionamos inimigos a lista

for i in range(enemy_num):
    
    enemies.append(turtle.Turtle())

for enemy in enemies:
    enemy.hideturtle()
    enemy.color("lime")
    enemy.shape("invader.gif")
    enemy.penup()
    enemy.speed(0)
    x = r.randint(-200,200)
    y = r.randint(100,250)
    enemy.setposition(x,y)
    enemy.showturtle()
enemyspd = 2

# Criar controles para o jogador

playerspd = 15

def move_left():
    x = player.xcor() # coordenada do objeto jogador em x
    x = x - playerspd
    if x < -280: # limita movimento maximo ate a borda
        x = -280
    player.setx(x) # define a nova posicao em x
    
def move_right():
    x = player.xcor()
    x = x + playerspd
    if x > 280:
        x = 280
    player.setx(x)

def fire():
    global bala_state # necessario, para caso a variavel seja alterada globalmente, e nao apenas dentro da funcao.

    if bala_state == "pronta":
        bala_state = "fire"
        x = player.xcor()
        y = player.ycor()
        bala.setposition(x,y + 20)
        bala.showturtle()

def isCollision(t1, t2): # definicao de colisao por pitagoras
    distance = math.sqrt(math.pow(t1.xcor() - t2.xcor(),2) + math.pow(t1.ycor() - t2.ycor(),2))
    if distance < 20:
        return True
    else:
        return False


# Cria a conexao com as teclas (arrow keys e space)
    
turtle.listen()
turtle.onkey(move_left, "Left")
turtle.onkey(move_right, "Right")
turtle.onkey(fire, "space")

# Programa principal do jogo

while True:
    
    if score % 250 == 0 and score != 0:
        enemies.append(turtle.Turtle())
        enemy = enemies[-1]
        enemy.hideturtle()
        enemy.color("lime")
        enemy.shape("invader.gif")
        enemy.penup()
        enemy.speed(0)
        x = r.randint(-200,200)
        y = r.randint(100,250)
        enemy.setposition(x,y)
        enemy.showturtle()
        score += 40
        scorestring = "Score: %d" %score
        score_pen.clear()
        score_pen.write(scorestring, False, align="left", font = ("Arial", 14, "normal"))
        if enemyspd < 0:
            enemyspd -= 1
        else:
            enemyspd += 1

    # Movemos nosso inimigo de um lado para o outro e tambem para baixo
    for enemy in enemies:
        x = enemy.xcor()
        x = x + enemyspd
        enemy.setx(x)
        
        if enemy.xcor() > 280 or enemy.xcor() < -280:
            for e in enemies:
                y = e.ycor()
                y = y - 40
                if y < -290:
                    y = -290
                e.sety(y)
            enemyspd *= -1
            
        #**
        if isCollision(bala, enemy):
            bala.hideturtle()
            bala_state = "pronta"
            bala.setposition(0, -400)
            if enemyspd < 0:
                x = r.randint(40,80)
                y = r.randint(40,80)
            else:
                x = r.randint(-80,-40)
                y = r.randint(40,80)
            x = enemy.xcor() + x
            y = enemy.ycor() + y
            if y > 250:
                y = 250
            if x > 200:
                x = 200
            if x < -200:
                x = -200
            enemy.setposition(x,y)
            score += 10
            scorestring = "Score: %d" %score
            score_pen.clear()
            score_pen.write(scorestring, False, align="left", font = ("Arial", 14, "normal"))

        if isCollision(player, enemy):
            player.hideturtle()
            enemy.hideturtle()
            bala.hideturtle()
            print("Game Over")
            end = True
            break
    if end == True:
        break

    # Movemos a bala quando necessario

    if bala_state == "fire":
        y = bala.ycor()
        y = y + balaspd
        bala.sety(y)

    # Deve interromper quando chegar ao topo

    if bala.ycor() > 275:
        if bala_state == "fire":
            bala.hideturtle()
            bala_state = "pronta"

    # Ou interrompe quando colidir com inimigo **

# Fim do jogo (press any key to exit)

input()


















