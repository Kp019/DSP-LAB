import numpy as np
import matplotlib.pyplot as plt

#impulse signal
'''
n=range(-5,5,1)
y = []

for i in range( len(n)):
    if n[i] == 0:
        temp = 1
    else:
        temp = 0
    
    y.append(temp)
print(n)
print(y)
plt.stem(n,y)
plt.axis
plt.show()

'''


#step signal
'''
n = range(-2,10,1)
y = []

for i in range(len(n)):
    if n[i] >= 0:
        temp = 1
    else:
        temp = 0
    y.append(temp)

print(n)
print(y)
plt.stem(n,y)
plt.show()

'''

#ramp signal
'''
n = range(-2,10,1)
y = []

for i in range(len(n)):
    if n[i] > 0 :
        temp = temp + 1
    else:
        temp = 0
    y.append(temp)

print(n)
print(y)
plt.plot(n,y)
plt.show()

'''


#triangular wave
'''
n = range(-5,5,1)
y = []

for i in range(len(n)):
    if n[i]%2 == 0:
        temp = 1
    else:
        temp = 0
    y.append(temp)
print(n)
print(y)
plt.plot(n,y)
plt.show()

'''

#bipolar pulse or square wave

n = range(-10,10,1)
y = []
i=0
t = len(n)

while i < t:    
    if n[i] %2 == 0:
        temp = 1
    else:
        temp = -1
    y.append(temp)
    i = i+1
    

print(n)
print(y)
plt.step(n,y)
plt.show()
