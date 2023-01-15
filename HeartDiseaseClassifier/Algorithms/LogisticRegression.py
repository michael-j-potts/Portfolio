import random
import numpy as np

def logistic_regression(x, y, iter = 100, lr = 0.01):
    n, m = len(x[0]), len(x)
    beta0, beta = initialize_params(n)
    for _ in range(iter):
        beta0grad, betagrad = compute_grad(x, y, beta0, beta, n, m, 50)
        beta0, beta = update_params(beta0, beta, beta0grad, betagrad, lr)
    return beta0, beta

def initialize_params(dims):
    beta0 = 0
    beta = [random.random() for _ in range(dims)]
    return beta0, beta

def compute_grad(x, y, beta0, beta, dims, m):
    beta0grad = 0
    betagrad = [0]*dims
    for i, point in enumerate(x):
        y_hat = logistic_function(point, beta0, beta)
        for j, feature in enumerate(point):
            betagrad[j] += (y_hat - y[i]) * feature / m
        beta0grad += (y_hat - y[i]) / m
    return beta0grad, betagrad

def logistic_function(point, beta0, beta):
    return 1 / (1 + np.exp(-(beta0 + point.dot(beta))))

def update_params(beta0, beta, beta0grad, betagrad, lr):
    beta0 += beta0grad * lr
    for i in range(len(beta)):
        beta[i] += (betagrad[i] * lr)
    return beta0, beta