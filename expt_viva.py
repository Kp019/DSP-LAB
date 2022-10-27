from matplotlib.cbook import flatten
import numpy as np
import matplotlib.pyplot as plt


def disp(x):
    with np.printoptions(precision=2, suppress=True):
        print(x)


x = np.array([[0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7]])

j = complex(0, 1)

# dft


def gendft(N):
    row = np.array([np.arange(N)])
    col = row.T
    f = col@row
    print(f)
    dft_matrix = np.exp(-j * 2*np.pi*f/N)
    disp(dft_matrix)
    return (dft_matrix)


N = 8
dft_matrix = gendft(N)
y = dft_matrix@x
disp(y)
