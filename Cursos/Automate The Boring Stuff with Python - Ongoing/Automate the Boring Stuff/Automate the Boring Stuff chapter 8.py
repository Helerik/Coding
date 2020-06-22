import os
import shelve
import random

print(os.path.join('usr','bin','spam')) #good for creating strings for filenames.
print()
myFiles = ['blah.txt', 'bloh.csv', 'bleh.exe']
for file in myFiles:
    print(os.path.join('C:\\Users\\defaultuser0', file)) #the backslashes need to be doubled for this to work.

print()
print(os.getcwd()) # Current Working Directory, i.e. gets the path to the folder which this script is saved in.
print()

tmp = os.getcwd()

os.chdir('C:\\Users') # changes this file to another directory
print(os.getcwd())

os.chdir(tmp)

# NOTE on diferent file paths:
## An absolute path, which always begins with the root folder
##
## A relative path, which is relative to the program’s current working directory

# .\ refers to a relative path directory.
#..\ refers to a relative path parent directory.

# Making new folder:
print()
try:
    tmp = os.path.join(os.getcwd(), "new_folder_for_testing")
    os.makedirs(tmp) # Creates a dir, at the specified path. Can make nested dirs at once.
except:
    print("This folder already exists!")

try:
    os.makedirs(os.path.join(tmp, "a_folder_inside_a_folder\\a_folder_inside_a_folder_inside_a_folder"))
except:
    print("Can't do that!")

print()

# Absolute and relative path handling:

print(os.path.abspath(tmp)) # finds the absolute path
print(os.path.isabs(tmp)) # is absolute path?
print(os.path.relpath(os.path.join(tmp, "a_folder_inside_a_folder\\a_folder_inside_a_folder_inside_a_folder"))) # os.path.relpath(path, start),
# where start is the relative parent directory. If ommited, start = current dir.

print()

path = tmp
print(os.path.basename(path))
print(os.path.dirname(path))
print(os.path.split(path))

# last one is direferent from .split method for a string

print(tmp.split(os.path.sep)) # finds each individual folder.

print()

print(os.path.basename(os.getcwd()), "folder has", os.path.getsize(os.getcwd()), "bytes.") # *** For some reason this is different than

print()
print(os.listdir(os.getcwd())) # all files in the 'path', ind this case, the cwdir.
print()

tmp = os.path.join(os.getcwd(), os.listdir(os.getcwd())[1])
print(os.path.basename(tmp), "has", os.path.getsize(tmp), "bytes.")
print()

summ = 0
for file in os.listdir(os.getcwd()):
    summ += os.path.getsize(os.path.join(os.getcwd(), file)) # *** this other method. Aparently the second one is the correct one.
print(summ)                                                  # might be a matter of compression....?

print()

print(os.path.exists(os.getcwd()))
print(os.path.exists('C:\\User'))
print(os.path.isfile(os.getcwd()))
print(os.path.isfile(tmp))
print(os.path.isdir(os.getcwd()))
print(os.path.isdir(tmp))

print()

# Verify if there is a disc (CD) on your computer:

for letter in range(65,91):
    if os.path.exists('%s:\\' %(chr(letter))):
        print('%s:\\' %(chr(letter)), "exists.")
    else:
        print('%s:\\' %(chr(letter)), "does not exist.")

# Handling files:
print()

file = open(os.path.join(os.getcwd(),"Hello.txt"), 'r')
print(file.read())
print()

file = open(os.path.join(os.getcwd(),"sonnet29.txt"), 'r')
print(file.readlines())
print()

baconFile = open("Bacon.txt", 'w')
baconFile.write("Hello world!\n")

baconFile = open("Bacon.txt", 'a')
baconFile.write("Bacon is not a vegetable.")

baconFile = open("Bacon.txt", 'r')
print(baconFile.read())
print()
# Shelve module: (for writing in binary)

shelfFile = shelve.open('mydata')
data = ["fii","fuu","foo"]
shelfFile['data'] = data # similar to a dictionary
shelfFile.close()

shelfFile = shelve.open('mydata')
print(type(shelfFile))
print(shelfFile['data'])
shelfFile.close()
print()
shelfFile = shelve.open('mydata')
print(list(shelfFile.keys())) # all my 'shelves' i.e. keys.

shelfFile['moData'] = ["bing", "bong"]
shelfFile.close()
shelfFile = shelve.open('mydata')
print(list(shelfFile.values())) # all values inside their keys
print()
print()

# Create random quiz generator:

# The quiz data. Keys are states and values are their capitals.
capitals = {'Alabama': 'Montgomery', 'Alaska': 'Juneau', 'Arizona': 'Phoenix',\
   'Arkansas': 'Little Rock', 'California': 'Sacramento', 'Colorado': 'Denver',\
   'Connecticut': 'Hartford', 'Delaware': 'Dover', 'Florida': 'Tallahassee',\
   'Georgia': 'Atlanta', 'Hawaii': 'Honolulu', 'Idaho': 'Boise', 'Illinois':\
   'Springfield', 'Indiana': 'Indianapolis', 'Iowa': 'Des Moines', 'Kansas':\
   'Topeka', 'Kentucky': 'Frankfort', 'Louisiana': 'Baton Rouge', 'Maine':\
   'Augusta', 'Maryland': 'Annapolis', 'Massachusetts': 'Boston', 'Michigan':\
   'Lansing', 'Minnesota': 'Saint Paul', 'Mississippi': 'Jackson', 'Missouri':\
   'Jefferson City', 'Montana': 'Helena', 'Nebraska': 'Lincoln', 'Nevada':\
   'Carson City', 'New Hampshire': 'Concord', 'New Jersey': 'Trenton', 'New Mexico':\
   'Santa Fe', 'New York': 'Albany', 'North Carolina': 'Raleigh',\
   'North Dakota': 'Bismarck', 'Ohio': 'Columbus', 'Oklahoma': 'Oklahoma City',\
   'Oregon': 'Salem', 'Pennsylvania': 'Harrisburg', 'Rhode Island': 'Providence',\
   'South Carolina': 'Columbia', 'South Dakota': 'Pierre', 'Tennessee':\
   'Nashville', 'Texas': 'Austin', 'Utah': 'Salt Lake City', 'Vermont':\
   'Montpelier', 'Virginia': 'Richmond', 'Washington': 'Olympia', 'West Virginia':\
   'Charleston', 'Wisconsin': 'Madison', 'Wyoming': 'Cheyenne'}

for num in range(35):
    quizFile = open('quiz%s.txt' %(1+num), 'w')
    answerFile = open('answer%s.txt' %(1+num), 'w')

    quizFile.write("Name:\n\nDate:\n\nPeriod:\n\n")
    quizFile.write(" "*20 + 'State Capitals Quiz (Form %s)\n\n' %(1+num))

    states = list(capitals.keys())
    random.shuffle(states)

    for qnum in range(50):
        quizFile = open('quiz%s.txt' %(1+num), 'a')
        answerFile = open('answer%s.txt' %(1+num), 'a')

        rightAnswer = capitals[states[qnum]]
        wrongAnswer = list(capitals.values())
        del(wrongAnswer[wrongAnswer.index(rightAnswer)])
        wrongAnswer = random.sample(wrongAnswer, 3)
        answerOptions = wrongAnswer + [rightAnswer]
        random.shuffle(answerOptions)

        quizFile.write('%s. What is the capital of %s?\n' %(qnum+1, states[qnum]))

        for i in range(4):
            quizFile.write(" %s. %s\n" %("ABCD"[i], answerOptions[i]))
        quizFile.write('\n')

        answerFile.write('%s. %s\n' % (qnum + 1, 'ABCD'[answerOptions.index(rightAnswer)]))
        quizFile.close()
        answerFile.close()
        
        
        
        




























































