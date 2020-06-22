
class Stack:

    def __init__(self, stack_list = []):
        if not isinstance(stack_list, list):
            print("Invalid argument type for Stack class.")
            stack_list = []
        self.items = stack_list

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)

    def __str__(self):
        ret_str = '['
        for i in range(len(self.items)):
            if i == len(self.items)-1:
                ret_str += str(self.items[i])
                break
            ret_str += str(self.items[i]) + ", "
        ret_str += ']'
        return ret_str
