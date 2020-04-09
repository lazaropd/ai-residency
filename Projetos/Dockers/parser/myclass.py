
class my_operations:

    def __init__(self, number):
        self.number = number

    def linear(self):
        return [i * self.number for i in range(10)]

    def quadratic(self):
        return [self.number ** i for i in range(10)]        
 
