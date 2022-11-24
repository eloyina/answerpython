import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Stack():
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[-1]
    def isEmpty(self):
        return self.items == []
    def size(self):
        return len(self.items)
    def __str__(self):
        return str(self.items)
    def __repr__(self):
        return str(self.items)

#una pila es una estructura de datos que funciona como una lista,
#  pero hace un pop para sacar el ultimo elemento agregado
pila= Stack()
pila.push(22)
pila.push(23)
pila.pop()
print(pila)

class Queue():
    def __init__(self):
        self.items = []
    def enqueue(self, item):
        self.items.insert(0,item)
    def dequeue(self):
        return self.items.pop()
    def isEmpty(self):
        return self.items == []
    def size(self):
        return len(self.items)
    def __str__(self):
        return str(self.items)
    def __repr__(self):
        return str(self.items)

#una cola es FIFO, first in first out primero
#  que entra primero que sale
cola= Queue()
cola.enqueue(12)
cola.enqueue(15)
cola.enqueue(1)

cola.dequeue()

print(cola)     

class Deque():
    def __init__(self):
        self.items = []
    def addFront(self, item):
        self.items.append(item)
    def addRear(self, item):
        self.items.insert(0,item)
    def removeFront(self):
        return self.items.pop()
    def removeRear(self):
        return self.items.pop(0)
    def isEmpty(self):
        return self.items == []
    def size(self):
        return len(self.items)
    def __str__(self):
        return str(self.items)
    def __repr__(self):
        return str(self.items)

#puedes agregarlo por todo lado
unicornio= Deque()
unicornio.addRear(10)
unicornio.addRear(41)
unicornio.addRear(49)
unicornio.addFront(1)
unicornio.removeFront()
print(unicornio)






