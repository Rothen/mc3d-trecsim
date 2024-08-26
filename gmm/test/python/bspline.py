import numpy as np
from mc3d_trecsim_gmm import BSpline, AUGMENTATION_MODE
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=30, linewidth=100000, precision=15)

# calculate the integral of a bspline basis function
def integral(bspline: BSpline, j: int, i: int, k: int):
    if k == 0:
        return float(i >= j and i < j+1)
    else:
        term1 = (bspline.getKnots()[i+k] - bspline.getKnots()[i]) / k * integral(bspline, j, i, k-1)
        term2 = (bspline.getKnots()[i+k+1] - bspline.getKnots()[i+1]) / k * integral(bspline, j, i+1, k-1)
        return term1 + term2

def integrate_bspline_basis_function(bspline: BSpline, i, k, a, b, num_points=1000):
    """
    Calculate the integral of the i-th basis function of a B-spline of degree k
    over the interval [a, b] using the trapezoidal rule.
    """
    if a >= b:
        return 0.0

    t_values = np.linspace(a, b, num_points)
    basis_values = [bspline.basis(t, i, k) for t in t_values]

    integral = np.trapz(basis_values, t_values)
    return integral

def int_again(spline: BSpline, j, k):
    return (spline.getKnots()[j+k+1] - spline.getKnots()[j]) / (k + 1)

def smooth_design_matrix(bspline: BSpline, k: int, A: np.ndarray, theta: float):
    offset = A.shape[0]
    A = np.vstack([A, np.zeros((bspline.getNumBasis()-2, bspline.getNumBasis()))])

    for j in range(2, bspline.getNumBasis()):
        N1 = integrate_bspline_basis_function(bspline, j, k-2, bspline.getKnots()[0], bspline.getKnots()[-1])
        print('N1:', N1)
        print('Nagain:', int_again(bspline, j, k-2))
        print('integral:', integral(bspline, j, j, k-2))
        s = N1**0.5
        d_j = (k-1)*(k-2)*s / (bspline.getKnots()[j+k-2] - bspline.getKnots()[j])
        b0 = d_j / (bspline.getKnots()[j+k-2] - bspline.getKnots()[j - 1])
        b2 = d_j / (bspline.getKnots()[j+k-1] - bspline.getKnots()[j])
        b1 = -(b0 + b2)
        A[offset+j-2, j-2:j+1] = theta*np.array([b0, b1, b2])
    
    return A

factor = 1000

x = np.array([0.0, 0.02, 0.06, 0.14, 0.17, 0.19, 0.22, 0.27, 0.33, 0.40, 0.47, 0.52, 0.56, 0.60, 0.62, 0.67, 0.75, 0.84, 0.92, 1.0]) * factor
y = np.array([0.15, 0.18, 0.20, 0.09, 0.19, 0.21, 0.18, 0.07, 0.20, 0.13, 0.17, 0.26, 0.19, 0.16, 0.23, 0.36, 0.29, 0.55, 0.59, 0.90]) * factor

plt.scatter(x, y, color='C4', alpha=0.5, label='Data')
degree = 3
knots = np.linspace(0, 1, 11) * factor
# knots = np.array([0, 0.2, 0.25, 0.3, 0.5, 0.55, 0.6, 0.8, 0.85, 0.9, 1.0]) * factor
bspline = BSpline(degree, AUGMENTATION_MODE.UNIFORM, knots)


##################### NO SMOOTHING #####################
A = bspline.designMatrix(x)
w, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
x_hat = np.linspace(0, 1, 1000) * factor
A = bspline.designMatrix(x_hat)
y_hat = A @ w

plt.plot(x_hat, y_hat, label='$\\theta=0$ (no smoothing)')

##################### SMOOTHING #####################

A = bspline.smoothDesignMatrix(bspline.designMatrix(x), 1e-3  * factor * 20)
y_pad = np.hstack([y, np.zeros((bspline.getNumBasis()-2))])
w, _, _, _ = np.linalg.lstsq(A, y_pad, rcond=None)
x_hat = np.linspace(0, 1, 1000) * factor
A = bspline.designMatrix(x_hat)
y_hat = A @ w

plt.plot(x_hat, y_hat, label='$\\theta=0.001$ (little smoothing)')

##################### SMOOTHING #####################
A = bspline.smoothDesignMatrix(bspline.designMatrix(x), 1e-2  * factor * 10)
y_pad = np.hstack([y, np.zeros((bspline.getNumBasis()-2))])
w, _, _, _ = np.linalg.lstsq(A, y_pad, rcond=None)
x_hat = np.linspace(0, 1, 1000) * factor
A = bspline.designMatrix(x_hat)
y_hat = A @ w

plt.plot(x_hat, y_hat, label='$\\theta=0.01$ (strong smoothing)')

'''##################### SMOOTHING #####################
A = bspline.designMatrix(x)
A = smooth_design_matrix(bspline, degree, A, 1e-2)
print(A)
# A = bspline.smoothDesignMatrix(bspline.designMatrix(x), 1e-2)
y_pad = np.hstack([y, np.zeros((bspline.getNumBasis()-2))])
w, _, _, _ = np.linalg.lstsq(A, y_pad, rcond=None)
x_hat = np.linspace(0, 1, 1000) * factor
A = bspline.designMatrix(x_hat)
y_hat = A @ w

plt.plot(x_hat, y_hat, label='$\\theta=0.01$')

##################### SMOOTHING #####################

A = bspline.designMatrix(x)
A = smooth_design_matrix(bspline, degree, A, 1e-3)
# A = bspline.smoothDesignMatrix(bspline.designMatrix(x), 1e-3)
y_pad = np.hstack([y, np.zeros((bspline.getNumBasis()-2))])
w, _, _, _ = np.linalg.lstsq(A, y_pad, rcond=None)
x_hat = np.linspace(0, 1, 1000) * factor
A = bspline.designMatrix(x_hat)
y_hat = A @ w

plt.plot(x_hat, y_hat, label='$\\theta=0.001$')'''

plt.legend()
plt.title('Comparison of Different Smoothing Parameters $\\theta$ for B-Splines')
plt.xlabel('x $[1]$')
plt.ylabel('y $[1]$')
plt.show()