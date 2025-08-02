from __future__ import division, print_function, unicode_literals
import numpy as np

# Thuật toán momentum
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new))/ len(theta_new) < 1e-3

def GD_momentum(theta_init, grad, eta, gamma):
    # Suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break
        theta.append(theta_new)
        v_old = v_new
    return theta

# Thuật toán Nesterov accelerated gradient (NAG)
def GD_NAG(w_init, grad, eta, gamma):
    w = [w_init]
    v = [np.zeros_like(w_init)]
    idx = 0
    for it in range(100):
        v_new = gamma*v[-1] + eta*grad(w[-1] - gamma*v[-1])
        w_new = w[-1] - v_new
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            idx = it
            break
        w.append(w_new)
        v.append(v_new)
    return w, idx

# Biến thể batch gradient descent
# Được hiểu là "Tất cả", tức là khi cập nhật theta = w, ta cần sử dụng tất cả các điểm dữ liệu x_i
# Điều này hạn chế đối với dữ liệu lớn ví dụ vài tỉ người dùng facebook


# Biến thể Stochastic Gradient Descent (SGD)
# Ví dụ Linear Regression theo SGD

# single point gradient
def sgrad(w, i, rd_id, Xbar, y):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, eta, X, Xbar, y):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id, Xbar, y)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:
                    return w
                w_last_check = w_this_check
    return w

# Biến thể Mini-batch Gradient Descent
# Khác với SGD, mini-batch sử dụng một số lượng n lớn hơn 1 (nhưng vẫn nhỏ hơn tổng số dữ liệu N rất nhiều).
# Giống với SGD, Mini-batch Gradient Descent bắt đầu mỗi epoch bằng việc xáo trộn ngẫu nhiên dữ liệu rồi chia toàn
# bộ dữ liệu thành các mini-batch, mỗi mini-batch có n điểm dữ liệu (trừ mini-batch cuối có thể có ít hơn nếu N
# không chia hết cho n).

# Stopping Criteria (Điều kiện dừng)

# Newton's Method
# 1: Áp dụng cho bài toán tìm nghiệm phương trình f(x) = 0
# f(x) = f'(x_t)(x - x_t) + f(x_t) => x_(t + 1) = x = x_t - f(x_t)/f'(x_t)
# 2: Áp dụng cho bài toán tìm local minimum: Áp dụng vào bài toán tìm nghiệm phương trình f'(x) = 0
# x_(t + 1) = x_t - (f''(x_t))^-1 . f'(x_t)