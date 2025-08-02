from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt


# GD cho hàm 1 biến:
# x_(t+1) = x_t - n.f'(x_t), n là learning rate, dấu - thể hiện đi ngược dấu đạo hàm
# Ví dụ f(x) = x^2 + 5sin(x)

def grad_gd1(x):  # f'(x)
    return 2 * x + 5 * np.cos(x)


def cost_gd1(x):  # f(x)
    return x ** 2 + 5 * np.sin(x)


def my_gd1(eta, x_0):
    x = [x_0]
    idx = 0
    for it in range(100):
        x_new = x[-1] - eta * grad_gd1(x[-1])
        if abs(grad_gd1(x_new)) < 1e-3:
            idx = it
            break
        x.append(x_new)
    return x, idx


def show_gd1():
    print("Show GD1:")

    (x1, it1) = my_gd1(0.1, -5)
    (x2, it2) = my_gd1(0.1, 5)
    print('Solution x1 = %f, cost = %f, obtained after %d iterations' % (x1[-1], cost_gd1(x1[-1]), it1))
    print('Solution x2 = %f, cost = %f, obtained after %d iterations' % (x2[-1], cost_gd1(x2[-1]), it2))


# GD cho hàm nhiều biến
# theta_(t+1) = theta_t - n.nabla_theta(f(theta_t)), theta là 1 vector, nabla là đạo hàm của hàm số tại 1 vector theta
# Linear Regression
# Ví dụ y = 4 + 3X

np.random.seed(2)
X = np.random.randn(1000, 1)
y = 4 + 3 * X + 0.2 * np.random.randn(1000, 1)  # noise added

# build Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.inv(A), b)

w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1 * x0

# Draw the fitting line
plt.plot(X.T, y.T, 'b.')  # data
plt.plot(x0, y0, 'y', linewidth=2)  # the fitting line
plt.axis((0, 1, 0, 10))
plt.show()

# Gradient loss function
def grad_gd2(W):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(W) - y)

# Loss function
def cost_gd2(W):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(W), 2)**2

def my_gd2(w_init, eta):
    W = [w_init]
    idx = 0
    for it in range(100):
        w_new = W[-1] - eta*grad_gd2(W[-1])
        if np.linalg.norm(grad_gd2(w_new))/len(w_new) < 1e-3:
            idx = it
            break
        W.append(w_new)
    return W, idx

def show_gd2():
    print("Show GD2:")

    w_init = np.array([[2], [1]])
    (w1, it1) = my_gd2(w_init, 1)
    print('Solution found by GD: w = ', w1[-1].T,',\n after %d iterations.' % (it1 + 1))

# Kiểm tra tính chính xác đạo hàm
def numerical_grad(W):
    eps = 1e-4
    g = np.zeros_like(W)
    for i in range(len(W)):
        w_p = W.copy()
        w_n = W.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost_gd2(w_p) - cost_gd2(w_n))/(2*eps)
    return g

def check_grad(W):
    grad1 = grad_gd2(W)
    grad2 = numerical_grad(W)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

if __name__ == '__main__':
    show_gd1()
    print()
    show_gd2()
    print()
    # Kiểm tra tính chính xác đạo hàm: Công thức xấp xỉ so với công thức Gradient Loss function
    W_check = np.random.rand(2, 1)
    print('Checking gradient...', check_grad(W_check))
