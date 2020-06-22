# Static Methods and Class Methods:

class Dog:

    # class variable, proper to this class. These are useful for static methods.
    # Basically a variable that is specific to the entire class of dogs. If we make a change to any dog, that affects this variable,
    # every other dog will have this change.
    dogs = []

    def __init__(self, name):
        self.name = name
        # this next line appends every dog we create to the list of dogs:
        self.dogs.append(self.name)

    # The following part we'll see decorators:

    # These are methods for the entire class and are necessary for taking variables such as 'dogs':
    @classmethod
    def num_dogs(cls):
        return len(cls.dogs)

    @staticmethod
    def bark(n):
        for _ in range(n):
            print("Bark!")

print(Dog.num_dogs())

d1 = Dog("TIM")
d2 = Dog("JIM")

# We can call the number of dogs from the class itself, instead of a specific dog.
print(len(Dog.dogs))
print(Dog.dogs)
print(len(d1.dogs))
print(Dog.num_dogs())
Dog.bark(2)
d1.bark(2)
print(d1.num_dogs())

print("\n"*5)


# for the next block, we'll look into private and public classes:

# Tipically, private classes can only be used in the file which it was created, while a public class can be used in whichever file.
# If it should be private, we use an _ (undersocre). It is just a convention though.

class _Private:
    def __init__(self, name):
        self.name = name

class notPrivate:
    def __init__(self,name):
        self.name = name

    # there can be private fuctions for a class
    def _display(self):
        print("Hello.")

    # and non-private functions inside the same class.
    def display(self):
        print("Hi.")
















