from tempfile import tempdir
import numpy as np
import matplotlib.pyplot as plt

def impulse():
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
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.axis([-5,5,0,2])
    plt.title("impulse")
    plt.show()

def step():
        
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
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("step signal")
    
    plt.axis([-2,10,0,2])
    plt.stem(n,y)
    plt.show()


def ramp():
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
    
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("ramp signal")    
    plt.axis([-2,10,0,10])
    plt.plot(n,y)
    plt.show()

def tri():
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
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("triangular signal")
    plt.axis([-5,5,0,2])

def bipolar():
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
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("bipolar signal")
    plt.axis([-10,10,-2,2])
    plt.step(n,y)
    plt.show()
    
def pulse():
    n = range(-5,5,1)
    y = []
    for i in range(len(n)):
        if abs(n[i]) < 2:
            temp = 1
        else:
            temp = 0
        y.append(temp)
    print(n)
    print(y)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("pulse signal")
    plt.axis([-5,5,0,2])
    plt.step(n,y)
    plt.show()

def cosine():
    t = np.linspace(0,1,1000)
    A = 2
    f = 5
    x = A * np.cos(2*np.pi*f*t)
    
    plt.xlabel("time")
    plt.ylabel("time")
    plt.title("cosine signal")
    plt.grid()
    plt.plot(t,x,'-b')
    plt.show()

def tri2():
    n = range(0,100,1)
    y= []

    for i in range(len(n)):
        r = n[i]%100
        if r <50:
            temp = r
        else:
            temp = 100-r
        y.append(temp)
        
    plt.plot(n,y)
    plt.axis([0,500,0,51])
    plt.xlabel("n--->")
    plt.ylabel("amplitude")
    plt.title("Triangular pulse")
    plt.grid()
    plt.show()

print("""1. impulse 
2. step() 
3. ramp() 
4. tri() 
5. bipolar()
6. pulse()
7. cosine()
8. tri2()
9. Exit""")

a=1

while( a == 1):
    j = int(input("enter the number"))
    if j == 1:
        impulse()
    elif j == 2:
        step()
    elif j == 3:
        ramp()
    elif j == 4:
        tri()
    elif j == 5:
        bipolar()
    elif j == 6:
        pulse()
    elif j == 7:
        cosine()
    elif j == 8:
        tri2()
    elif j == 9:
        a=0
    else:
        print("enter the valid number")