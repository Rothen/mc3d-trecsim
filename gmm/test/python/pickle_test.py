from mc3d_trecsim_gmm import Camera, Frame, GMMParam, LBFGSParam
import pickle
import numpy as np

camera = Camera("")
frame = Frame(camera, [], 1000, 8000)
gmmParam = GMMParam()
lbfgsParam = LBFGSParam()

print('#############################################################')
print('Camera:', camera)

print('id:', camera.id)
print('A:', camera.A)
print('d:', camera.d)
print('P:', camera.P)

camera.id = "something"
camera.A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * -1
camera.d = np.array([9, 8, 7, 6, 5])
camera.P = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]) * -1
camera.height = 500
camera.width = 1000

print('#############################################################')
print('Frame:', frame)

frame.camera.id = "something"
frame.camera.A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) * -2
frame.camera.d = np.array([9, 8, 7, 6, 5]) * 2
frame.camera.P = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]) * -2
frame.camera.height = 5
frame.camera.width = 10

print('#############################################################')
print('GMMParam:')

print('KEYPOINTS:', gmmParam.KEYPOINTS)
print('nu:', gmmParam.nu)
print('maxIter:', gmmParam.maxIter)
print('keypointConfidenceThreshold:', gmmParam.keypointConfidenceThreshold)
print('tol:', gmmParam.tol)
print('splineDegree:', gmmParam.splineDegree)
print('splineKnotDelta:', gmmParam.splineKnotDelta)
print('maxFrameBuffer:', gmmParam.maxFrameBuffer)

gmmParam.KEYPOINTS += [0, 1, 2]
gmmParam.nu += 1
gmmParam.maxIter += 1
gmmParam.keypointConfidenceThreshold += 1
gmmParam.tol += 1
gmmParam.splineDegree += 1
gmmParam.splineKnotDelta += 1
gmmParam.maxFrameBuffer += 1

print('#############################################################')
print('LBFGSParam:')

print('m:', lbfgsParam.m)
print('epsilon:', lbfgsParam.epsilon)
print('epsilon_rel:', lbfgsParam.epsilon_rel)
print('past:', lbfgsParam.past)
print('delta:', lbfgsParam.delta)
print('max_iterations:', lbfgsParam.max_iterations)
print('linesearch:', lbfgsParam.linesearch)
print('max_linesearch:', lbfgsParam.max_linesearch)
print('min_step:', lbfgsParam.min_step)
print('max_step:', lbfgsParam.max_step)
print('ftol:', lbfgsParam.ftol)
print('wolfe:', lbfgsParam.wolfe)

lbfgsParam.m += 1
lbfgsParam.epsilon += 1
lbfgsParam.epsilon_rel += 1
lbfgsParam.past += 1
lbfgsParam.delta += 1
lbfgsParam.max_iterations += 1
lbfgsParam.linesearch += 1
lbfgsParam.max_linesearch += 1
lbfgsParam.min_step += 1
lbfgsParam.max_step += 1
lbfgsParam.ftol += 1
lbfgsParam.wolfe += 1

print('#############################################################')




with open("camera.obj","wb") as f:
    pickle.dump(camera, f)

with open("frame.obj","wb") as f:
    pickle.dump(frame, f)

with open("gmmParam.obj","wb") as f:
    pickle.dump(gmmParam, f)

with open("lbfgsParam.obj","wb") as f:
    pickle.dump(lbfgsParam, f)



print('#############################################################')
with open("camera.obj",'rb') as f:
    camera = pickle.load(f)
    print('Reconstructed Camera:', camera)
    print('id:', camera.id)
    print('A:', camera.A)
    print('d:', camera.d)
    print('P:', camera.P)

print('#############################################################')
with open("frame.obj",'rb') as f:
    frame = pickle.load(f)
    camera = frame.camera
    print('Reconstructed Frame:', frame)
    print('id:', camera.id)
    print('A:', camera.A)
    print('d:', camera.d)
    print('P:', camera.P)

print('#############################################################')
with open("gmmParam.obj",'rb') as f:
    recGmmParam = pickle.load(f)
    print('Reconstructed GMMParam:', recGmmParam)
    print('KEYPOINTS:', recGmmParam.KEYPOINTS)
    print('nu:', recGmmParam.nu)
    print('maxIter:', recGmmParam.maxIter)
    print('keypointConfidenceThreshold:', recGmmParam.keypointConfidenceThreshold)
    print('tol:', recGmmParam.tol)
    print('splineDegree:', recGmmParam.splineDegree)
    print('splineKnotDelta:', recGmmParam.splineKnotDelta)
    print('maxFrameBuffer:', recGmmParam.maxFrameBuffer)

print('#############################################################')
with open("lbfgsParam.obj",'rb') as f:
    lbfgsParam = pickle.load(f)
    print('Reconstructed LBFGSParam:', lbfgsParam)
    print('m:', lbfgsParam.m)
    print('epsilon:', lbfgsParam.epsilon)
    print('epsilon_rel:', lbfgsParam.epsilon_rel)
    print('past:', lbfgsParam.past)
    print('delta:', lbfgsParam.delta)
    print('max_iterations:', lbfgsParam.max_iterations)
    print('linesearch:', lbfgsParam.linesearch)
    print('max_linesearch:', lbfgsParam.max_linesearch)
    print('min_step:', lbfgsParam.min_step)
    print('max_step:', lbfgsParam.max_step)
    print('ftol:', lbfgsParam.ftol)
    print('wolfe:', lbfgsParam.wolfe)

print('#############################################################')