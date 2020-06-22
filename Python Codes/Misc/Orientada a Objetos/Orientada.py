# Code for learning Object Oriented Programming in python 3.xx

# The following code shows how to create a simple object class and a function which serves this object
# The object can have properties which we define.
class Robot:

    def introduce_self(self):
        print("My name is " + self.name)

r1 = Robot()
r1.name = "Tom"
r1.color = "red"
r1.weight = 30

r1.introduce_self()

# In the prior example, the object atributtes are defined outside of the object class.
# For this reason, we may use a constructor:

class Robot_:

    # We define fixed object properties. It must take self as Its first atributte.
    def __init__(self, name, color, weight):
        self.name = name
        self.color = color
        self.weight = weight

    # This function is what we call a method
    def introduce_self(self):
        print("My name is " + self.name)

r2 = Robot_("Bob", "blue", 25)

r2.introduce_self()
# In a general manner, the piece of code above is cleaner and simpler/more compact than the previous one.

#besides that, we can also call an objects specific attribute:
print(r2.color)
print(r1.weight - r2.weight)

print("\n"*5)




class Person:

    def __init__(self, name, personality, sitting, robot):
        self.name = name
        self.personality = personality
        self.isSitting = sitting
        self.robot_owned = robot

    def sit_down(self):
        self.isSitting = True

    def sit_up(self):
        self.isSitting = False
        
    
p1 = Person("Paula", "Funny", True, r2)

print(p1.name)
p1.robot_owned.introduce_self()
print("\n"*5)

# It is possible to have classes and objects interacting with each other

# For the next part, we can see a method, called inheritance:

class Dog:

    def __init__(self, name, age):

        self.name = name
        self.age = age

    def speak(self):

        print("Hi, I am", self.name, "and I am", self.age, "years old.")

# The following code creates a derived/parent-child class. The Cat is taken from Dog.
class Cat(Dog):

    def __init__(self, name, age, color):

        super().__init__(name, age)
        self.color = color

    def talk(self):
        print("Meow")

c1 = Cat("Tim", 5, "brown")

c1.speak()
c1.talk()

# The super() command takes the properties from the dog class and gives it to the cat class.

print("\n"*5)

# Overloading methods:
# This consists in creating new operations, specific to an object

class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
        self.coord = (self.x, self.y)

    def move(self, x, y):
        self.x += x
        self.y += y

    def length(self):
        from math import sqrt
        return sqrt(self.x**2 + self.y**2)

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __mul__(self, p):
        return self.x*p.x + self.y*p.y

    def __gt__(self, p ):
        return self.length() > p.length()
        
    def __ge__(self, p ):
        return self.length() >= p.length()

    def __lt__(self, p ):
        return self.length() < p.length()

    def __le__(self, p ):
        return self.length() <= p.length()

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    # Method for printing (it has to be a string):
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    # __len__ defines the length of an object.
    # There are many other built-in methods in python such as the above

p1 = Point(1,2)
p2 = Point(3,4)
print(p1+p2)
print(p1*p2)
print(p1 == p2)
print(p1.length())
print(p2.length())
print(p1 >= p2)
print(p1 == p1)

















