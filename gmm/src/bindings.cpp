#include "bspline.h"
#include "gmm.h"
#include "mc3d_common.h"
#include "em.h"
#include "gmm_container.h"
#include <LBFGS.h>
#include <string>
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace MC3D_TRECSIM;
using BSplineD = BSpline<double>;
using VectorD = Vector<double>;
using RowMatrixD = RowMatrix<double>;
using ColMatrixD = ColMatrix<double>;
using IntrinsicMatrixD = IntrinsicMatrix<double>;
using DistortionVectorD = DistortionVector<double>;
using ExtrinsicMatrixD = ExtrinsicMatrix<double>;
using WorldPointsD = WorldPoints<double>;
using WorldPointD = WorldPoint<double>;
using CameraPointsD = CameraPoints<double>;
using CameraD = Camera<double>;
using GMMD = GMM<double>;
using GMMContainerD = GMMContainer<double>;
using GMMParamD = GMMParam<double>;
using FrameD = Frame<double>;
using LBFGSParamD = LBFGSpp::LBFGSParam<double>;
using GMMParametersD = GMMParameters<double>;
using EMFitResultD = EMFitResult<double>;
using MC3DModelD = MC3DModel<double>;
using MultivariateNormalD = MultivariateNormal<double, 2>;

PYBIND11_MODULE(gmm, m)
{
    py::enum_<AUGMENTATION_MODE>(m, "AUGMENTATION_MODE")
        .value("UNIFORM", AUGMENTATION_MODE::UNIFORM)
        .value("SAME", AUGMENTATION_MODE::SAME)
        .value("NONE", AUGMENTATION_MODE::NONE)
        .export_values();

    py::enum_<LBFGSpp::LINE_SEARCH_TERMINATION_CONDITION>(m, "LINE_SEARCH_TERMINATION_CONDITION", "The enumeration of line search termination conditions.")
        .value("LBFGS_LINESEARCH_BACKTRACKING_ARMIJO", LBFGSpp::LINE_SEARCH_TERMINATION_CONDITION::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO, "Backtracking method with the Armijo condition.The backtracking method finds the step length such that it satisfiesthe sufficient decrease (Armijo) condition,\\f$f(x + a \\cdot d) \\le f(x) + \\beta' \\cdot a \\cdot g(x)^T d\\f$,where \\f$x\\f$ is the current point, \\f$d\\f$ is the current search direction,\\f$a\\f$ is the step length, and \\f$\\beta'\\f$ is the value specified by\\ref LBFGSParam::ftol. \\f$f\\f$ and \\f$g\\f$ are the functionand gradient values respectively.")
        .value("LBFGS_LINESEARCH_BACKTRACKING", LBFGSpp::LINE_SEARCH_TERMINATION_CONDITION::LBFGS_LINESEARCH_BACKTRACKING, "The backtracking method with the defualt (regular Wolfe) condition. An alias of `LBFGS_LINESEARCH_BACKTRACKING_WOLFE`.")
        .value("LBFGS_LINESEARCH_BACKTRACKING_WOLFE", LBFGSpp::LINE_SEARCH_TERMINATION_CONDITION::LBFGS_LINESEARCH_BACKTRACKING_WOLFE, "Backtracking method with regular Wolfe condition. The backtracking method finds the step length such that it satisfies both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`) and the curvature condition, \\f$g(x + a \\cdot d)^T d \\ge \\beta \\cdot g(x)^T d\\f$, where \\f$\\beta\\f$ is the value specified by \\ref LBFGSParam::wolfe.")
        .value("LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE", LBFGSpp::LINE_SEARCH_TERMINATION_CONDITION::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE, "Backtracking method with strong Wolfe condition. The backtracking method finds the step length such that it satisfies both the Armijo condition (`LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`) and the following condition, \\f$\\vert g(x + a \\cdot d)^T d\\vert \\le \\beta \\cdot \\vert g(x)^T d\\vert\\f$, where \\f$\\beta\\f$ is the value specified by \\ref LBFGSParam::wolfe.")
        .export_values();

    py::class_<GMMParametersD>(m, "GMMParameters")
        .def(py::init<>())
        .def_readwrite("theta", &GMMParametersD::theta)
        .def_readwrite("pi", &GMMParametersD::pi);

    py::class_<MultivariateNormalD>(m, "MultivariateNormal")
        .def(py::init<>());

    py::class_<EMFitResultD>(m, "EMFitResult")
        .def_readonly("parameters", &EMFitResultD::parameters)
        .def_readonly("designMatrix", &EMFitResultD::designMatrix)
        .def_readonly("responsibilities", &EMFitResultD::responsibilities)
        .def_readonly("diff", &EMFitResultD::diff)
        .def_readonly("convergence", &EMFitResultD::convergence)
        .def_readonly("niters", &EMFitResultD::niters);

    py::class_<BSplineD>(m, "BSpline")
        .def(py::init<unsigned int, AUGMENTATION_MODE, VectorD>(),
             py::arg("degree"),
             py::arg("augmentationMode"),
             py::arg("knots"))
        .def("getKnots", &BSplineD::getKnots)
        .def("getNumBasis", &BSplineD::getNumBasis)
        .def("getDegree", &BSplineD::getDegree)
        .def("basis", &BSplineD::basis,
             py::arg("t"),
             py::arg("j"),
             py::arg("k"))
        .def("basisGrad", &BSplineD::basisGrad,
             py::arg("t"),
             py::arg("j"),
             py::arg("k"))
        .def("basisInt", &BSplineD::basisInt,
             py::arg("j"),
             py::arg("k"))
        .def("designMatrix", py::overload_cast<const VectorD &>(&BSplineD::designMatrix),
             py::arg("t"))
        .def("designMatrix", py::overload_cast<const VectorD &, RowMatrixD &>(&BSplineD::designMatrix),
             py::arg("t"),
             py::arg("A"))
        .def(
            "smoothDesignMatrix", [](BSplineD &bspline, RowMatrixD &designMatrix, double lambda)
            {
                bspline.smoothDesignMatrix(designMatrix, lambda);
                return designMatrix; },
            py::arg("designMatrix"),
            py::arg("smoothingFactor")) // cannot be named lambda because it is a reserved keyword in python
        .def("pushKnot", &BSplineD::pushKnot,
             py::arg("knot"))
        .def("popKnotFront", &BSplineD::popKnotFront);

    py::class_<CameraD>(m, "Camera")
        .def(py::init<const std::string &>())
        .def_readwrite("id", &CameraD::id)
        .def_readwrite("height", &CameraD::height)
        .def_readwrite("width", &CameraD::width)
        .def_readwrite("distance", &CameraD::distance)
        .def_readwrite("A", &CameraD::A)
        .def_readwrite("Ainv", &CameraD::Ainv)
        .def_readwrite("d", &CameraD::d)
        .def_readwrite("P", &CameraD::P)
        .def_readwrite("R", &CameraD::R)
        .def_readwrite("RT", &CameraD::RT)
        .def_readwrite("t", &CameraD::t)
        .def_property_readonly("shape", [](const CameraD &camera)
                               { return py::make_tuple(camera.height, camera.width); })
        .def(
            "setCalibration",
            [](CameraD &camera, IntrinsicMatrixD A, DistortionVectorD d, ExtrinsicMatrixD P, unsigned int height, unsigned int width, double distance)
            {
                IntrinsicMatrixD _A = A;
                DistortionVectorD _d = d;
                ExtrinsicMatrixD _P = P;
                camera.setCalibration(A, d, P, height, width, distance);
            },
            py::arg("A"),
            py::arg("d"),
            py::arg("P"),
            py::arg("height"),
            py::arg("width"),
            py::arg("distance") = double(1.0))
        .def("toCameraCoordinates", [](CameraD &camera, WorldPointsD &PWs)
             { return camera.toCameraCoordinates(PWs); })
        .def("pixelsToWorldPoints", [](CameraD &camera, CameraPointsD &pIs, double distance)
             { return camera.pixelsToWorldPoints(pIs, distance); })
        .def("pixelsToWorldPoints", [](CameraD &camera, CameraPointsD &pIs)
             { return camera.pixelsToWorldPoints(pIs); })
        .def("toCameraCoordinatesSingle", [](CameraD &camera, WorldPointD &PW)
             { return camera.toCameraCoordinates(PW); })
        .def("project", [](CameraD &camera, WorldPointsD &PWs)
             { return camera.project(PWs); })
        .def("projectSingle", [](CameraD &camera, WorldPointD &PW)
             { return camera.projectSingle(PW); })
        .def("projectGrad", [](CameraD &camera, WorldPointD &PW)
             { return camera.projectGrad(PW); })
        .def("transformWorldCenter", &CameraD::transformWorldCenter)
        .def(py::pickle(
            [](const CameraD &camera)
            {
                return py::make_tuple(camera.id, camera.A, camera.d, camera.P, camera.height, camera.width);
            },
            [](py::tuple cameraTuple)
            {
                CameraD camera = CameraD(cameraTuple[0].cast<std::string>());

                IntrinsicMatrixD A = cameraTuple[1].cast<IntrinsicMatrixD>();
                DistortionVectorD d = cameraTuple[2].cast<DistortionVectorD>();
                ExtrinsicMatrixD P = cameraTuple[3].cast<ExtrinsicMatrixD>();

                camera.setCalibration(A, d, P, cameraTuple[4].cast<unsigned int>(), cameraTuple[5].cast<unsigned int>(), double(1.0));

                return camera;
            }));

    py::class_<FrameD>(m, "Frame")
        .def(py::init<size_t, std::vector<RowMatrixD> &, double, double>(),
             py::arg("cameraIndex"),
             py::arg("kpts"),
             py::arg("time"),
             py::arg("origTimestamp"))
        .def_readwrite("cameraIndex", &FrameD::cameraIndex)
        .def_readwrite("kpts", &FrameD::kpts)
        .def_readwrite("time", &FrameD::time)
        .def_readwrite("origTimestamp", &FrameD::origTimestamp)
        .def(py::pickle(
            [](const FrameD &frame)
            {
                return py::make_tuple(frame.cameraIndex, frame.kpts, frame.time, frame.origTimestamp);
            },
            [](py::tuple frameTuple)
            {
                size_t cameraIndex = frameTuple[0].cast<size_t>();
                std::vector<RowMatrixD> kpts = frameTuple[1].cast<std::vector<RowMatrixD>>();

                FrameD frame = FrameD(cameraIndex, kpts, frameTuple[2].cast<double>(), frameTuple[3].cast<double>());

                return frame;
            }));

    py::class_<GMMParamD>(m, "GMMParam")
        .def(py::init<>())
        .def_readwrite("KEYPOINTS", &GMMParamD::KEYPOINTS)
        .def_readwrite("nu", &GMMParamD::nu)
        .def_readwrite("maxIter", &GMMParamD::maxIter)
        .def_readwrite("keypointConfidenceThreshold", &GMMParamD::keypointConfidenceThreshold)
        .def_readwrite("tol", &GMMParamD::tol)
        .def_readwrite("splineDegree", &GMMParamD::splineDegree)
        .def_readwrite("maxFrameBuffer", &GMMParamD::maxFrameBuffer)
        .def_readwrite("splineKnotDelta", &GMMParamD::splineKnotDelta)
        .def_readwrite("autoManageTheta", &GMMParamD::autoManageTheta)
        .def_readwrite("autoManageHypothesis", &GMMParamD::autoManageHypothesis)
        .def_readwrite("copyLastThetas", &GMMParamD::copyLastThetas)
        .def_readwrite("splineSmoothingFactor", &GMMParamD::splineSmoothingFactor)
        .def_readwrite("numSupportCameras", &GMMParamD::numSupportCameras)
        .def_readwrite("notSupportedSinceThreshold", &GMMParamD::notSupportedSinceThreshold)
        .def_readwrite("responsibilityLookback", &GMMParamD::responsibilityLookback)
        .def_readwrite("responsibilitySupportThreshold", &GMMParamD::responsibilitySupportThreshold)
        .def_readwrite("totalResponsibilitySupportThreshold", &GMMParamD::totalResponsibilitySupportThreshold)
        .def_readwrite("dragAlongUnsupportedKeyPoints", &GMMParamD::dragAlongUnsupportedKeyPoints)
        .def_readwrite("minValidKeyPoints", &GMMParamD::minValidKeyPoints)
        .def("setSeed", &GMMParamD::setSeed)
        .def("getSeed", &GMMParamD::getSeed)
        .def(py::pickle(
            [](const GMMParamD &gmmParam)
            {
                return py::make_tuple(
                    gmmParam.KEYPOINTS,
                    gmmParam.nu,
                    gmmParam.maxIter,
                    gmmParam.keypointConfidenceThreshold,
                    gmmParam.tol,
                    gmmParam.splineDegree,
                    gmmParam.splineKnotDelta,
                    gmmParam.maxFrameBuffer,
                    gmmParam.autoManageTheta,
                    gmmParam.copyLastThetas,
                    gmmParam.splineSmoothingFactor,
                    gmmParam.autoManageHypothesis,
                    gmmParam.numSupportCameras,
                    gmmParam.responsibilitySupportThreshold,
                    gmmParam.notSupportedSinceThreshold,
                    gmmParam.responsibilityLookback,
                    gmmParam.totalResponsibilitySupportThreshold,
                    gmmParam.dragAlongUnsupportedKeyPoints,
                    gmmParam.minValidKeyPoints);
            },
            [](py::tuple gmmParamTuple)
            {
                GMMParamD gmmParam{};

                gmmParam.KEYPOINTS = gmmParamTuple[0].cast<std::vector<int>>();
                gmmParam.nu = gmmParamTuple[1].cast<double>();
                gmmParam.maxIter = gmmParamTuple[2].cast<unsigned int>();
                gmmParam.keypointConfidenceThreshold = gmmParamTuple[3].cast<double>();
                gmmParam.tol = gmmParamTuple[4].cast<double>();
                gmmParam.splineDegree = gmmParamTuple[5].cast<double>();
                gmmParam.splineKnotDelta = gmmParamTuple[6].cast<double>();
                gmmParam.maxFrameBuffer = gmmParamTuple[7].cast<unsigned int>();
                gmmParam.autoManageTheta = gmmParamTuple[8].cast<bool>();
                gmmParam.copyLastThetas = gmmParamTuple[9].cast<bool>();
                gmmParam.splineSmoothingFactor = gmmParamTuple[10].cast<double>();
                gmmParam.autoManageHypothesis = gmmParamTuple[11].cast<bool>();
                gmmParam.numSupportCameras = gmmParamTuple[12].cast<size_t>();
                gmmParam.notSupportedSinceThreshold = gmmParamTuple[13].cast<size_t>();
                gmmParam.responsibilityLookback = gmmParamTuple[14].cast<size_t>();
                gmmParam.responsibilitySupportThreshold = gmmParamTuple[15].cast<double>();
                gmmParam.totalResponsibilitySupportThreshold = gmmParamTuple[16].cast<double>();
                gmmParam.dragAlongUnsupportedKeyPoints = gmmParamTuple[17].cast<bool>();
                gmmParam.minValidKeyPoints = gmmParamTuple[18].cast<int>();

                return gmmParam;
            }));

    py::class_<LBFGSParamD>(m, "LBFGSParam")
        .def(py::init<>())
        .def_readwrite("m", &LBFGSParamD::m, "The number of corrections to approximate the inverse Hessian matrix. The L-BFGS routine stores the computation results of previous \\ref m iterations to approximate the inverse Hessian matrix of the current iteration. This parameter controls the size of the limited memories (corrections). The default value is \\c 6. Values less than \\c 3 are not recommended. Large values will result in excessive computing time.")
        .def_readwrite("epsilon", &LBFGSParamD::epsilon, "Absolute tolerance for convergence test. This parameter determines the absolute accuracy \\f$\\epsilon_{abs}\\f$ with which the solution is to be found. A minimization terminates when \\f$||g|| < \\max\\{\\epsilon_{abs}, \\epsilon_{rel}||x||\\}\\f$, where \f$||\\cdot||\\f$ denotes the Euclidean (L2) norm. The default value is \\c 1e-5.")
        .def_readwrite("epsilon_rel", &LBFGSParamD::epsilon_rel, "Relative tolerance for convergence test. This parameter determines the relative accuracy \\f$\\epsilon_{rel}\\f$ with which the solution is to be found. A minimization terminates when \\f$||g|| < \\max\\{\\epsilon_{abs}, \\epsilon_{rel}||x||\\}\\f$, where \\f$||\\cdot||\\f$ denotes the Euclidean (L2) norm. The default value is \\c 1e-5.")
        .def_readwrite("past", &LBFGSParamD::past, "Distance for delta-based convergence test. This parameter determines the distance \\f$d\\f$ to compute the rate of decrease of the objective function, \\f$f_{k-d}(x)-f_k(x)\\f$, where \\f$k\\f$ is the current iteration step. If the value of this parameter is zero, the delta-based convergence test will not be performed. The default value is \\c 0.")
        .def_readwrite("delta", &LBFGSParamD::delta, "Delta for convergence test. The algorithm stops when the following condition is met, \\f$|f_{k-d}(x)-f_k(x)|<\\delta\\cdot\\max(1, |f_k(x)|, |f_{k-d}(x)|)\\f$, where \\f$f_k(x)\\f$ is the current function value, and \\f$f_{k-d}(x)\\f$ is the function value \\f$d\\f$ iterations ago (specified by the \\ref past parameter). The default value is \\c 0.")
        .def_readwrite("max_iterations", &LBFGSParamD::max_iterations, "The maximum number of iterations. The optimization process is terminated when the iteration count exceeds this parameter. Setting this parameter to zero continues an optimization process until a convergence or error. The default value is \\c 0.")
        .def_readwrite("linesearch", &LBFGSParamD::linesearch, "The line search termination condition. This parameter specifies the line search termination condition that will be used by the LBFGS routine. The default value is `LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE`.")
        .def_readwrite("max_linesearch", &LBFGSParamD::max_linesearch, "The maximum number of trials for the line search. This parameter controls the number of function and gradients evaluations per iteration for the line search routine. The default value is \\c 20.")
        .def_readwrite("min_step", &LBFGSParamD::min_step, "The minimum step length allowed in the line search. The default value is \\c 1e-20. Usually this value does not need to be modified.")
        .def_readwrite("max_step", &LBFGSParamD::max_step, "The maximum step length allowed in the line search. The default value is \\c 1e+20. Usually this value does not need to be modified.")
        .def_readwrite("ftol", &LBFGSParamD::ftol, "A parameter to control the accuracy of the line search routine. The default value is \\c 1e-4. This parameter should be greater than zero and smaller than \\c 0.5.")
        .def_readwrite("wolfe", &LBFGSParamD::wolfe, "The coefficient for the Wolfe condition. This parameter is valid only when the line-search algorithm is used with the Wolfe condition. The default value is \\c 0.9. This parameter should be greater the \ref ftol parameter and smaller than \\c 1.0.")
        .def(py::pickle(
            [](const LBFGSParamD &lbfgsParam)
            {
                return py::make_tuple(lbfgsParam.m, lbfgsParam.epsilon, lbfgsParam.epsilon_rel, lbfgsParam.past, lbfgsParam.delta, lbfgsParam.max_iterations, lbfgsParam.linesearch, lbfgsParam.max_linesearch, lbfgsParam.min_step, lbfgsParam.max_step, lbfgsParam.ftol, lbfgsParam.wolfe);
            },
            [](py::tuple lbfgsParamTuple)
            {
                LBFGSParamD lbfgsParam{};

                lbfgsParam.m = lbfgsParamTuple[0].cast<int>();
                lbfgsParam.epsilon = lbfgsParamTuple[1].cast<double>();
                lbfgsParam.epsilon_rel = lbfgsParamTuple[2].cast<double>();
                lbfgsParam.past = lbfgsParamTuple[3].cast<int>();
                lbfgsParam.delta = lbfgsParamTuple[4].cast<double>();
                lbfgsParam.max_iterations = lbfgsParamTuple[5].cast<int>();
                lbfgsParam.linesearch = lbfgsParamTuple[6].cast<int>();
                lbfgsParam.max_linesearch = lbfgsParamTuple[7].cast<int>();
                lbfgsParam.min_step = lbfgsParamTuple[8].cast<double>();
                lbfgsParam.max_step = lbfgsParamTuple[6].cast<double>();
                lbfgsParam.ftol = lbfgsParamTuple[7].cast<double>();
                lbfgsParam.wolfe = lbfgsParamTuple[8].cast<double>();

                return lbfgsParam;
            }));

    py::class_<GMMContainerD>(m, "GMMContainer")
        .def(py::init<>())
        .def(py::init<int, int, std::vector<CameraD>&, double, const RowMatrixD &>(),
             py::arg("KEYPOINT"),
             py::arg("J"),
             py::arg("cameras"),
             py::arg("nu"),
             py::arg("designMatrix"))
        .def_readwrite("parameters", &GMMContainerD::parameters);

    py::class_<GMMD>(m, "GMM")
        .def(py::init<int, std::vector<CameraD>, GMMParamD, LBFGSParamD>(),
             py::arg("J"),
             py::arg("cameras"),
             py::arg("gmmParam"),
             py::arg("lbfgsParam"))
        .def_readonly("spline", &GMMD::spline)
        .def_readonly("designMatrix", &GMMD::designMatrix)
        .def_readonly("gmmContainers", &GMMD::gmmContainers)
        .def_readonly("frames", &GMMD::frames)
        .def_readonly("J", &GMMD::J)
        .def_readonly("gmmParam", &GMMD::gmmParam)
        .def_readonly("cameras", &GMMD::cameras)
        .def(
            "fit",
            [](GMMD &gmm, const std::map<int, ColMatrixD> &initialThetas, const std::map<int, VectorD> &initialPis)
            {
                std::map<int, EMFitResultD> fitResults = gmm.fit(initialThetas, initialPis);
                py::dict pyFitResults;

                for (auto fitResult : fitResults)
                {
                    py::dict pyFitResult(
                        "theta"_a = fitResult.second.parameters.theta,
                        "pi"_a = fitResult.second.parameters.pi,
                        "designMatrix"_a = fitResult.second.designMatrix,
                        "diff"_a = fitResult.second.diff,
                        "convergence"_a = fitResult.second.convergence,
                        "responsibilities"_a = fitResult.second.responsibilities,
                        "niters"_a = fitResult.second.niters,
                        "supports"_a = fitResult.second.supports
                    );
                    pyFitResults[py::cast(fitResult.first)] = pyFitResult;
                }

                return pyFitResults;
            },
            py::arg("initialThetas") = std::map<int, ColMatrixD>(),
            py::arg("initialPis") = std::map<int, VectorD>(),
            py::return_value_policy::reference_internal,
            "Fit the frames to a 3D spline")
        .def("addFrame", &GMMD::addFrame,
             py::arg("frame"))
        .def("__lshift__", [](GMMD &self, FrameD &frame)
             { return self << frame; });

    m.doc() = "MC3D-TRECSIM c++ implementation.";
}
