
# Parses a mathematical expression with operations (+, -, *, /) and returns the result of the expression
def parse_to_expression(string):

    num1 = ''
    num2 = ''

    sign = ''
    if string[0] == '-':
        sign = '-'
        string_tmp = ''
        i = 0
        while i < len(string):
            if string[i] == '+':
                string_tmp += '-'
            elif string[i] == '-':
                if i > 0:
                    string_tmp += '+'
            elif string[i] == '(' and string[i+1] in "1234567890":
                string_tmp += "(-"
            elif string[i] == '(' and string[i+1] == '-':
                string_tmp += '('
                i += 1
            else:
                string_tmp += string[i]
            i += 1
        string = string_tmp
    
    for i in range(len(string)):

        # While the parser detects digits or dot (.), saves the number.
        if string[i] in "1234567890.":
            num1 += string[i]

        # If the parser detects the start of a parenthesis, it looks for it
        if string[i] == '(':
            par_counter = 1
            for j in range(i+1, len(string)):
                if string[j] == '(':
                    par_counter += 1
                if string[j] == ')':
                    par_counter -= 1
                if par_counter == 0:
                    break
                
            # Once the parser detects the end of the parenthesis, the first number is
            # assigned to be the result of the operation inside the parenthesis.
            # After that, the parser appends this result to the rest of the string and parses it
            num1 = parse_to_expression(string[i+1:j])
            num2 = parse_to_expression(num1 + string[j+1:])

            return sign + num2

        if string[i] == '+':
            num2 = parse_to_expression(string[i+1:])
            num1 = str(float(num1) + float(num2))
            return num1

        if string[i] == '-':
            num2 = parse_to_expression(string[i:])
            num1 = str(float(num1) - float(num2))
            return sign + num1

        if string[i] == '*':
            if string[i+1] == '(':
                par_counter = 1
                for j in range(i+2, len(string)):
                    if string[j] == '(':
                        par_counter += 1
                    if string[j] == ')':
                        par_counter -= 1
                    if par_counter == 0:
                        break
                num2 = parse_to_expression(string[i+2:j])
                string = str(float(num1) * float(num2)) + string[j+1:]
                return parse_to_expression(string)
            
            for j in range(i+1,len(string)):
                if string[j] in "+-":
                    break
                if j == len(string)-1:
                    j = len(string)
                    break
                
            num2 = parse_to_expression(string[i+1:j])
            num1 = str(float(num1) * float(num2))
            num1 = parse_to_expression(num1 + string[j:])
            
            return num1

        if string[i] == '/':
            if string[i+1] == '(':
                par_counter = 1
                for j in range(i+2, len(string)):
                    if string[j] == '(':
                        par_counter += 1
                    if string[j] == ')':
                        par_counter -= 1
                    if par_counter == 0:
                        break
                num2 = parse_to_expression(string[i+2:j])
                string = str(float(num1) / float(num2)) + string[j+1:]
                return parse_to_expression(string)
            
            for j in range(i+1,len(string)):
                if string[j] in "+-":
                    break
                if j == len(string)-1:
                    j = len(string)
                    break
                
            num2 = parse_to_expression(string[i+1:j])
            num1 = str(float(num1) / float(num2))
            num1 = parse_to_expression(num1 + string[j:])

            return num1

    return num1

string = "3-9-2*2-(1+2)"
print(parse_to_expression(string))
print(eval(string))

print('\n///\n')

string = "(3*2)-2*2+(-2)*(1+2)"
print(parse_to_expression(string))
print(eval(string))

print('\n///\n')


string = "3-9-2*2-(-1)*(1+2+1)"
print(parse_to_expression(string))
print(eval(string))



        








        
