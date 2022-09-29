import scipy.fftpack as fftpack
import cmath
import numpy as np
import matplotlib.pyplot as plt

'''part - A
*******************************************************************************************************'''


'''
N = 4
j = complex(0,-1)
w = cmath.exp(j*2*cmath.pi/N)
j,k= np.meshgrid(np.arange ( N ),np.arange( N ))
dft = np.power(w,j*k)
dft = np.real(dft)
plt.imshow(dft)
plt.show()

x = [1,1,0,0] #random i/p matrix

#loop for finding DFT

for i in range(N):
    l = 0
    y = 0
    while l < N:
        y = y + x[l] + w**l*i
        l = l+1
    print(y,",")

'''


#part - B
'''****************************************************************************************************************'''

'''MAX_SIZE = 10

# Function to find circular convolution


def convolution(x, h, n, m):
    row_vec = [0] * MAX_SIZE
    col_vec = [0] * MAX_SIZE
    out = [0] * MAX_SIZE
    circular_shift_mat = [[0 for i in range(MAX_SIZE)]
                          for j in range(MAX_SIZE)]

    # Finding the maximum size between the
    # two input sequence sizes
    if (n > m):
        maxSize = n
    else:
        maxSize = m

    # Copying elements of x to row_vec and padding
    # zeros if size of x < maxSize
    for i in range(maxSize):
        if (i >= n):
            row_vec[i] = 0
        else:
            row_vec[i] = x[i]

    # Copying elements of h to col_vec and padding
    # zeros if size of h is less than maxSize
    for i in range(maxSize):
        if (i >= m):
            col_vec[i] = 0
        else:
            col_vec[i] = h[i]

    # Generating 2D matrix of
    # circularly shifted elements
    k = 0
    d = 0

    for i in range(maxSize):
        curIndex = k - d
        for j in range(maxSize):
            circular_shift_mat[j][i] = \
                row_vec[curIndex % maxSize]
            curIndex += 1

        k = maxSize
        d += 1

    # Computing result by matrix
    # multiplication and printing results
    for i in range(maxSize):
        for j in range(maxSize):
            out[i] += circular_shift_mat[i][j] * \
                col_vec[j]

        print(out[i], end=" ")


# Driver program
if __name__ == '__main__':
    x = [1, 2, 3, 4]
    n = len(x)
    h = [2, 3, 4]
    m = len(h)

    convolution(x, h, n, m)
'''


'''Part - c
***************************************************************************************************'''


pi = np.pi

tdata = np.arange(5999.)/300
dt = tdata[1]-tdata[0]

datay = np.sin(pi*tdata)+2*np.sin(pi*2*tdata)
N = len(datay)

fouriery = abs(fftpack.rfft(datay))/N

freqs = fftpack.rfftfreq(len(datay), d=(tdata[1]-tdata[0]))

df = freqs[1] - freqs[0]

parceval = sum(datay**2)*dt - sum(fouriery**2)*df
print(parceval)

plt.plot(freqs, fouriery, 'b-')
plt.xlim(0, 3)
plt.show()
