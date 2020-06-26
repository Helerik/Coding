from StackClass import Stack
import numpy as np

class Expression_Parser():

    def __is_number(self, string):
        try:
            string = float(string)
            return True
        except:
            return False

    def __prepare(self, string):
        string = string.replace(' ', '')
        string_tmp = ''
        for i in range(len(string)):
            if string[i] in "0123456789.":
                string_tmp += string[i]
            elif string[i] in "abcdefghijklmnopqrstuvwxyz":
                string_tmp += string[i]
            elif string[i] == '^':
                string_tmp += " ^ "
            elif string[i] == '*':
                string_tmp += " * "
            elif string[i] == '/':
                string_tmp += " / "
            elif string[i] == '+':
                string_tmp += " + "
            elif string[i] == '-':
                if i == 0 or (i > 0 and string[i-1] == '('):
                    string_tmp += ' 0 - '
                else:
                    string_tmp += ' - '
            elif string[i] == '(':
                string_tmp += ' ( '
            else:
                string_tmp += ' ) '
        return string_tmp

    def postfix_from(self, infix):
        infix = self.__prepare(infix)
        prec = {}
        prec["^"] = 4
        prec["*"] = 3
        prec["/"] = 3
        prec["+"] = 2
        prec["-"] = 2
        prec["("] = 1
        opStack = Stack()
        postfixList = []
        tokenList = infix.split()

        for token in tokenList:
            if self.__is_number(token) or token == "pi" or token == "e":
                postfixList.append(token)
            elif token == '(':
                opStack.push(token)
            elif token == ')':
                topToken = opStack.pop()
                while topToken != '(':
                    postfixList.append(topToken)
                    topToken = opStack.pop()
            else:
                while (not opStack.isEmpty()) and \
                   (prec[opStack.peek()] >= prec[token]):
                      postfixList.append(opStack.pop())
                opStack.push(token)

        while not opStack.isEmpty():
            postfixList.append(opStack.pop())
        return " ".join(postfixList)

    def Evaluate(self, postfix):
        opStack = Stack()
        tokenList = postfix.split()

        for token in tokenList:
            if self.__is_number(token) or token == "pi" or token == "e":
                if token == "pi":
                    token = np.pi
                elif token == "e":
                    token = np.e
                opStack.push(float(token))
            else:
                op2 = opStack.pop()
                op1 = opStack.pop()
                result = self.__doMath(token,op1,op2)
                opStack.push(result)
        return opStack.pop()

    def __doMath(self, op, op1, op2):
        if op == "^":
            return op1 ** op2
        elif op == "*":
            return op1 * op2
        elif op == "/":
            return op1 / op2
        elif op == "+":
            return op1 + op2
        elif op == "-":
            return op1 - op2















