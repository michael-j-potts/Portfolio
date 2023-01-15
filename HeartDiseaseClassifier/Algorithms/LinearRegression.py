import random

def linear_regression(x, y, iter = 100, lr = 0.01):
    n, m = len(x[0]), len(x)
    beta0, beta = initialize_params(n)
    for _ in range(iter):
        beta0grad, betagrad = compute_grad(x, y, beta0, beta, n, m)
        beta0, beta = update_params(beta0, beta, beta0grad, betagrad, lr)
    return beta0, beta

def initialize_params(dims):
    beta0 = 0
    beta = [random.random() for _ in range(dims)]
    return beta0, beta

def compute_grad(x, y, beta0, beta, dims, m):
    beta0grad = 0
    betagrad = [0]*dims
    for i in range(m):
        y_hat = sum(x[i][j]*beta[j] for j in range(dims)) + beta0
        error = 2 * (y[i] - y_hat)
        for j in range(dims):
            betagrad[j] += error * x[i][j] / m
        beta0grad += error / m
    return beta0grad, betagrad

def update_params(beta0, beta, beta0grad, betagrad, lr):
    beta0 -= beta0grad * lr
    for i in range(len(beta)):
        beta[i] -= (betagrad[i] * lr)
    return beta0, beta

