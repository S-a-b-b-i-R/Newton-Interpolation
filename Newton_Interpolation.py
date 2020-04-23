import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


def get_diff_table(X, Y):

    n = len(X)
    A = np.zeros([n, n])

    for i in range(0, n):
        A[i][0] = Y[i]

    for j in range(1, n):
        for i in range(j, n):
            A[i][j] = (A[i][j - 1] - A[i - 1][j - 1]) / (X[i] - X[i - j])

    return A


def newton_interpolation(X, Y, x):

    sum = Y[0]
    temp = np.zeros((len(X), len(X)))

    for i in range(0, len(X)):
        temp[i, 0] = Y[i]
    temp_sum = 1.0
    for i in range(1, len(X)):

        temp_sum = temp_sum * (x - X[i - 1])

        for j in range(i, len(X)):
            temp[j, i] = (temp[j, i - 1] - temp[j - 1, i - 1]) / (X[j] - X[j - i])
        sum += temp_sum * temp[i, i]
    return sum


X = [-1, 0, 1, 2, 3, 4, 5]
Y = [-20, -12, 1, 15, 4, 21, 41]
A = get_diff_table(X, Y)
df = pd.DataFrame(A)
df

xs = np.linspace(np.min(X), np.max(X), 1000, endpoint=True)
ys = []
for x in xs:
    ys.append((newton_interpolation(X, Y, x)))

plt.title("Newton Interpolation")
plt.plot(X, Y, 's', label="original values")
plt.plot(xs, ys, 'r', label="interpolation values")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)
plt.show()
plt.savefig("newton_interpolation.png", bbox_inches='tight')
