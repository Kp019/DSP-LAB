from re import I
from matplotlib.cbook import flatten
import numpy as np
import matplotlib.pyplot as plt
import time


def disp(x):
    with np.printoptions(precision=2, suppress=True):
        print(x)


x = np.array([[1],
              [2],
              [3],
              [4]])

j = complex(0, 1)

# dft


def gendft(N):
    row = np.array([np.arange(4)])
    col = row.T
    f = col@row
    print(f)

    j = complex(0, 1)
    dft_matrix = np.exp(-j * 2*np.pi*f/N)
    disp(dft_matrix)
    return (dft_matrix)


N = 4
dft_matrix = gendft(N)
y = dft_matrix@x
disp(y)


# fft
fft1 = np.fft.fft(x.flatten())
disp(fft1)
plt.stem(np.abs(y.flatten()-fft1))
plt.ylim([-0.5, 0.5])
plt.show()


value = np.arange(2, 12+1, 1)
dft_mtx_time = np.zeros(len(value))
fft_routine_time = np.zeros(len(value))
idx = 0

for v in value:
    n = 2**v
    col_data = np.random.rand(N, 1)
    y = gendft(n)
    start_time = time.perf_counter()
    fft_frm_dftmtx = y@col_data
    end_time = time.perf_counter()
    dft_mtx_time[idx] = end_time - start_time

    start_time = time.perf_counter()
    fft_frm_dftmtx = y@col_data
    end_time = time.perf_counter()
    fft_routine_time[idx] = end_time - start_time
    idx = idx+1
    fft_routine_time

plt.plot(value, dft_mtx_time, 'r', value, fft_routine_time, 'k')
plt.show()

# circular Convolution

n1 = 3
n2 = 5
x1 = np.random.randint(6, size=n1)
x2 = np.random.randint(6, size=n2)

y = np.convolve(x1, x2)
M = max(n1, n2)

yc = y[0:M]
overlap_idx = np.arange(len(y[M:]))
print(x1)
print(x2)
print(yc)
plt.subplot(3, 3, 1)
plt.stem(x1)
plt.title("sequance 1")

plt.subplot(3, 3, 2)
plt.stem(x2)
plt.title("sequance 2")

plt.subplot(3, 3, 3)
plt.stem(yc)
plt.title("Circular Convolution")
plt.tight_layout()
plt.show()


# parsevals theorem
M = 5000
x1 = np.random.rand(1, M) + np.random.rand(1, M) * 1j
x2 = np.random.rand(1, M) + np.random.rand(1, M) * 1j

# timedomain
TD_inner_prod = np.sum(x1*x2.conj())
print(TD_inner_prod)

# freq domain
x1_fft = np.fft.fft(x1)
x2_fft = np.fft.fft(x2)
FD_inner_prod = np.sum(x1_fft*x2_fft.conj())/M
print(FD_inner_prod)
