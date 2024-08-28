import matplotlib
import numpy as np
import sys
from calibration import Calibration
import pickle
from enums import KPT_IDXS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gmm
import matplotlib.pyplot as plt
from .gmm import MC3DModel, GMMContainer, Camera, BSpline, AUGMENTATION_MODE, GMMParam
from typing import Any, Callable

print(float(1000 * np.finfo(np.float32).eps))
exit()

##############################################################################################
def central_diff(fn: Callable, diff_arg: int = 0, args: list[Any] = [], h: float = float(1000 * np.finfo(np.float32).eps)):
    args[diff_arg] += h
    x1 = fn(*args)
    args[diff_arg] -= 2*h
    x2 = fn(*args)

    return (x1 - x2) / (2*h)
###############################################################################################

###################################### TEST x^3 ###############################################
'''finite_diffs = []
analytic_diffs = []

xs = np.random.uniform(-10.0, 10.0, 1000)

for x in xs:
    finite_diff = central_diff(lambda xn: xn**3, args=[x])
    analytic_diff = 3 * x**2
    
    finite_diffs.append(finite_diff)
    analytic_diffs.append(analytic_diff)

finite_diffs = np.array(finite_diffs)
analytic_diffs = np.array(analytic_diffs)

plt.figure()
plt.title('||Analytical Derivative - Finite Difference|| of $\\frac{d}{d x}x^3$')
plt.plot(analytic_diffs - finite_diffs)
plt.grid()
plt.show()'''
###############################################################################################

###################################### TEST MC3DModel ###############################################
'''bspline = BSpline(np.linspace(0, 1, 3), 3, AUGMENTATION_MODE.UNIFORM)

responsibilities = np.array([])                                         # RowMatrixD
pi = np.array([1])                                                      # VectorD
keypoint = 5                                                            # int
J = 1                                                                   # int
gmmParam = GMMParam()                                                   # GMMParamD
gmmParam.splineSmoothingFactor = 10000

gmmContainer = GMMContainer(bspline, keypoint, J)                       # GMMContainerD
designMatrix = bspline.designMatrix(np.array(np.linspace(0, 1, 10)))    # RowMatrixD
designMatrix = bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor)

hGrads = []                                                             # std::vector<RowMatrixD>
for r in range(designMatrix.shape[0]):
    designMatrixRow = designMatrix[r]
    hGrad = np.zeros((3, bspline.getNumBasis() * 3))
    hGrad[0, :bspline.getNumBasis()] = designMatrixRow
    hGrad[1, bspline.getNumBasis():2 * bspline.getNumBasis()] = designMatrixRow
    hGrad[2, 2 * bspline.getNumBasis():] = designMatrixRow
    hGrads.append(hGrad)

nu = 1e-2                                                               # double

def fn(_theta):
    numBasis = bspline.getNumBasis()
    PWs = designMatrix @ _theta
    offset = designMatrix.shape[0] - numBasis + 2
    
    fx = 0
    thetaGrad = np.zeros((numBasis*3, ))    
    
    for n in range(numBasis-2):
        for j in range(J):
            PW = PWs[offset + n, j*3:j*3+3]
            fx += np.dot(PW, PW)
            PWGrad = hGrads[offset + n]
            thetaGrad[j * numBasis * 3 : (j * numBasis * 3) + numBasis * 3] += (2 * gmmParam.splineSmoothingFactor * PW.reshape((1, 3)) @ PWGrad).flatten()
    return fx, thetaGrad

finite_diffs = []
analytic_diffs = []

xs = np.random.uniform(-10.0, 10.0, 1000)

for x in xs:
    theta = np.random.uniform(-1.0, 1.0, (bspline.getNumBasis(), J*3)) * 50

    finite_diff = central_diff(lambda _t: fn(_t)[0], args=[theta])
    analytic_diff = fn(theta)[1].sum()
    
    finite_diffs.append(finite_diff)
    analytic_diffs.append(analytic_diff)

finite_diffs = np.array(finite_diffs)
analytic_diffs = np.array(analytic_diffs)

plt.figure()
plt.title('||Analytical Derivative - Finite Difference|| of $\\frac{d}{d x}x^3$')
plt.plot(analytic_diffs - finite_diffs)
plt.grid()
plt.show()'''
###############################################################################################

###################################### TEST MC3DModel ###############################################
bspline = BSpline(np.linspace(0, 1, 3), 3, AUGMENTATION_MODE.UNIFORM)

responsibilities = np.array([])                                         # RowMatrixD
pi = np.array([1])                                                      # VectorD
keypoint = 5                                                            # int
J = 1                                                                   # int
gmmParam = GMMParam()                                                   # GMMParamD
gmmParam.splineSmoothingFactor = 10

gmmContainer = GMMContainer(bspline, keypoint, J)                       # GMMContainerD
designMatrix = bspline.designMatrix(np.array(np.linspace(0, 1, 10)))    # RowMatrixD
designMatrix = bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor)

hGrads = []                                                             # std::vector<RowMatrixD>
for r in range(designMatrix.shape[0]):
    designMatrixRow = designMatrix[r]
    hGrad = np.zeros((3, bspline.getNumBasis() * 3))
    hGrad[0, :bspline.getNumBasis()] = designMatrixRow
    hGrad[1, bspline.getNumBasis():2 * bspline.getNumBasis()] = designMatrixRow
    hGrad[2, 2 * bspline.getNumBasis():] = designMatrixRow
    hGrads.append(hGrad)

nu = 1e-2                                                               # double

def fn(_theta):
    numBasis = bspline.getNumBasis()
    PWs = designMatrix @ _theta
    offset = designMatrix.shape[0] - numBasis + 2
    
    fx = 0
    thetaGrad = np.zeros((numBasis*3, ))    
    
    for n in range(numBasis-2):
        for j in range(J):
            PW = PWs[offset + n, j*3:j*3+3]
            fx += np.dot(PW, PW)
            PWGrad = hGrads[offset + n]
            thetaGrad[j * numBasis * 3 : (j * numBasis * 3) + numBasis * 3] += (2 * PW.reshape((1, 3)) @ PWGrad).flatten()
    return fx, thetaGrad

finite_diffs = []
analytic_diffs = []

xs = np.random.uniform(-10.0, 10.0, 1000)

def central_diff(fn: Callable, diff_arg: int = 0, args: list[Any] = [], h: float = float(1000 * np.finfo(np.float32).eps), i: int = 0, j: int = 0):
    args[diff_arg][i, j] += h
    x1 = fn(*args)
    args[diff_arg][i, j] -= 2*h
    x2 = fn(*args)

    return (x1 - x2) / (2*h)
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

for x in xs:
    theta = np.random.uniform(-1.0, 1.0, (bspline.getNumBasis(), J*3)) * 50
    finite_diff = np.zeros((bspline.getNumBasis(), J*3))
    for i in range(bspline.getNumBasis()):
        for j in range(J*3):
            finite_diff[i, j] = central_diff(lambda _t: fn(_t)[0], args=[theta], i=i, j=j)
    analytic_diff = fn(theta)[1]
    finite_diff = finite_diff.T.flatten()
    finite_diffs.append(finite_diff)
    analytic_diffs.append(analytic_diff)
    
    print(analytic_diff)
    print(finite_diff)

finite_diffs = np.array(finite_diffs)
analytic_diffs = np.array(analytic_diffs)

# plt.figure()
# plt.title('||Analytical Derivative - Finite Difference|| of $\\frac{d}{d x}x^3$')
# plt.plot(analytic_diffs - finite_diffs)
# plt.grid()
# plt.show()