from mc3d_trecsim_gmm import AUGMENTATION_MODE, BSpline
import numpy as np

AUGMENTATION_MODE = AUGMENTATION_MODE.SAME
spline = BSpline(np.array([1, 2, 3]), 3, AUGMENTATION_MODE)
print('SAME:\n', spline.getKnots(), '\n')
spline.popKnotFront()
print('SAME:\n', spline.getKnots(), '\n')