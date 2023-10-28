import random
import numpy as np
from sympy import *
import time

x = symbols('x')
y = symbols('y')

# Functions
fx = 2 * x ** 2 + 3 * x + 1
gx = 1 - 0.6 * exp(-x**2 - y**2) - 0.4 * exp(-(x + 1.75)**2 - (y - 1)**2)
hx = (x+2*y-7)**2 + (2*x+y-5)**2

grad_fx = diff(fx, x)  # Automatic gradient of function fx
grad_gx_x = diff(gx, x)  # Automatic gradient of function gx with respect to x
grad_gx_y = diff(gx, y)  # Automatic gradient of function gx with respect to y
grad_hx_x = diff(hx, x)  # Automatic gradient of function hx with respect to x
grad_hx_y = diff(hx, y)  # Automatic gradient of function hx with respect to y

def funcf(x1):
    return fx.subs(x, x1)

def funcg(x1, y1):
    return gx.subs({x: x1, y: y1})

def funch(x1, y1):
    return gx.subs({x: x1, y: y1})

def gradient_descent(f, grad_x, grad_y, ptx, pty):
    """
    Perform gradient descent optimization for a function.

    Parameters:
    f (function): The function to optimize (fx or gx).
    grad_x (expression): Gradient with respect to x.
    grad_y (expression): Gradient with respect to y.
    ptx (float): Initial x-coordinate.
    pty (float): Initial y-coordinate.

    Returns:
    None
    """
    lr = 0.1  # learning rate
    precision = 1e-12  # precision of found minimum
    max_iterations = 10000  # maximum number of iterations

    iterations = 0  # starting number of iterations

    for i in range(max_iterations):
        current_x = ptx
        current_y = pty
        if f == fx:
            ptx = current_x - lr * grad_x.subs(x, current_x)  # update x point for fx
        else:
            ptx = current_x - lr * grad_x.subs({x: current_x, y: current_y})  # update x point for gx
            pty = current_y - lr * grad_y.subs({x: current_x, y: current_y})  # update y point for gx
        step_x = ptx - current_x
        step_y = pty - current_y
        iterations += 1
        if abs(step_x) <= precision and abs(step_y) <= precision:
            break

    if f == fx:
        print("Function f(x):")
        print(f"Minimum at x: {ptx} y: {funcf(ptx)}")
    elif f == gx:
        print("Function g(x, y):")
        print(f"Minimum at x: {ptx} y: {pty} z: {funcg(ptx, pty)}")
    else:
        print("Function h(x, y):")
        print(f"Minimum at x: {ptx} y: {pty} z: {funch(ptx, pty)}")
    print(f"Number of iterations: {iterations}")

start_time = time.time()

# Parameters
point_x = round(random.uniform(-7, 7), 2)  # x position of the starting point
point_y = round(random.uniform(-7, 7), 2)  # y position of the starting point
print(f"Starting point at: x = {point_x}, y = {point_y}")

gradient_descent(fx, grad_fx, None, point_x, point_y)
gradient_descent(gx, grad_gx_x, grad_gx_y, point_x, point_y)
gradient_descent(hx, grad_hx_x, grad_hx_y, point_x, point_y)

end_time = time.time()
whole_time = (end_time - start_time)

print(f"Elapsed time: {whole_time * 1000} ms")
