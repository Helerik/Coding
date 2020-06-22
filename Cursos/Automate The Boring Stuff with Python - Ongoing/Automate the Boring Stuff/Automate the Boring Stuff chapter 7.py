import re
import pyperclip

phoneNumRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
mo = phoneNumRegex.search("My phone number is 321-456-8910, and hers is 123-345-1242")
print("Phone number found: " + mo.group())
    
# the regex module is used for finding certain patterns or words is a string.

print()

def find(word, sentence):
    wordRegex = re.compile(word)
    mo = wordRegex.search(sentence)

    try:
        print("Found: " + mo.group())
    except:
        print("Found: none")

find("hello", "hello world, hello")

# to find parenthesis, use \( and \)

print()

batRegex = re.compile(r'Bat(man|mobile|copter|bat)')
mo = batRegex.search('Batmobile lost a wheel')
print(mo.group())

mo = batRegex.search(r'batcopter and Batman')
print(mo.group())

mo = batRegex.search(r'Batcopter and Batman')
print(mo.group())

print()

batRegex = re.compile(r'Bat(wo)?man')
mo1 = batRegex.search('The Adventures of Batman')
print(mo1.group())

mo2 = batRegex.search('The Adventures of Batwoman')
print(mo2.group())


# findall method:

print()

phoneNumRegex = re.compile(r'(\d\d\d-\d\d\d-\d\d\d\d)|(\d\d\d\d-\d\d\d\d)')
mo = phoneNumRegex.findall("My phone number is 321-456-8910, and hers is 123-345-1242, 8389-1283")
for i in range(len(mo)):
    for word in mo[i]:
        if word == '':
            pass
        else:
            print("Phone number found:", word)

print()

# case sensitive:

robocop = re.compile(r'robocop', re.IGNORECASE)
mo = robocop.findall("rOboCOp is a robocOP")
for word in mo:
    print(word.lower())

# string substitution

print()

namesRegex = re.compile(r'(\w+)')
print(namesRegex.sub('*********', 'sad789JKKO'))

print()

# Project: Phone Number and E-Mail extractor:

phoneRegex = re.compile(r'''(
                                (\d{3}|\(\d{3}\))?        # area code
                                (\s|-|\.)?                  # separator
                                (\d{3}|\(\d{3}\))                     # first 3 digits
                                (\s|-|\.)
                                (\d{3})
                                (\s|-|\.)
                                (\d{4})
                                (\s*(ext|x|ext)\s*\d{2,5})? # extension
                                )''', re.VERBOSE)

emailRegex = re.compile(r'''(
                                [a-zA-Z0-9._%+-]+           # email personal code
                                @                           # at
                                [a-zA-Z0-9.-]+              # email service name
                                (\.com)                     # dot com
                                (\.\w{2,3})?                # country complement
                                )''', re.VERBOSE)

text = str(pyperclip.paste())

matches = []

for group in phoneRegex.findall(text):
    if not group[0].strip() in matches:
        matches.append(group[0].strip())
for group in emailRegex.findall(text):
    if group[0].strip() not in matches:
        matches.append(group[0].strip())

if matches != []:
    print("Found from clipboard:")
    for found in matches:
        print(found)

print()






















