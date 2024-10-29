import sys
import numpy as np

sys.path.append('../../build')
from gmm import BSpline, AUGMENTATION_MODE

AUGMENTATION_MODE = AUGMENTATION_MODE.SAME
spline = BSpline(np.array([1, 2, 3]), 3, AUGMENTATION_MODE)
print('SAME:\n', spline.getKnots(), '\n')
spline.popKnotFront()
print('SAME:\n', spline.getKnots(), '\n')