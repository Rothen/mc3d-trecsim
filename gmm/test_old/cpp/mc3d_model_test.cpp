#include "mc3d_common.h"
#include "gmm.h"
#include "bspline.h"
#include "bspline.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "gmm_param.h"
#include "test_shared.h"
#include "em.h"
#include "gmm_maximizer.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iomanip>
#include <memory>

using namespace MC3D_TRECSIM;

TEST(MC3DModel, Call)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::vector<int> KEYPOINTS = {5};
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    double keypointConfidenceThreshold = 0.5;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    gmm.prepareCalculations();

    ColMatrix<double> theta(3, 5);
    theta << -1.254598811526375002e+01, 4.507143064099161478e+01, 2.319939418114050866e+01, 9.865848419703659999e+00, -3.439813595575634508e+01, -3.440054796637973311e+01, -4.419163878318005345e+01, 3.661761457749351933e+01, 1.011150117432088003e+01, 2.080725777960454792e+01, -4.794155057041975709e+01, 4.699098521619943369e+01, 3.324426408004217137e+01, -2.876608893217238361e+01, -3.181750327928993727e+01;

    gmm.gmmContainers[5].parameters.theta = theta.transpose();
    for (Support &support : gmm.gmmContainers[5].supports)
    {
        support.supported = true;
    }

    // spline, designMatrix, hGrads, cameras, gmmParam, lbfgsParam
    GMMMaximizer<double> maximizer(gmm.spline, gmm.designMatrix, gmm.hGrads, gmm.cameras, gmm.gmmParam, gmm.lbfgsParam);
    EM<double> em(gmmParam.maxIter, gmmParam.tol, maximizer);

    auto resp = em.expectation(gmm.gmmContainers[5]);
    Vector<double> pi(1);
    pi << 1;

    MC3DModel<double> fun(gmm.spline, resp, gmm.gmmContainers[5], gmm.hGrads, cameras);
    Vector<double> x = gmm.gmmContainers[5].parameters.theta.reshaped().transpose();
    Vector<double> thetaGrad = Vector<double>::Zero(x.size());

    double fx = fun(x, thetaGrad);
    double expectedFx = 73930026.288746863604;

    Vector<double> expectedThetaGrad(15);
    expectedThetaGrad << 5869.8611128789445, 97237.9042186133802, 1371761.2896209229802, 1334152.6610209115461, 59287.0434321050766, -5659.5141689859190, 69378.3996834269487, 1926100.5819975740451, 1907956.7754413790681, 86690.5386007088623, -14439.3097098863223, -1131290.3777934416212, -23641616.4393547223881, -23288179.4390413648216, -987407.6688519464369;

    std::cout << "Responsibilities: ";
    std::cout << resp.format(HeavyFormat) << std::endl;
    std::cout << "Function Value: ";
    std::cout << std::fixed << std::setprecision(12) << fx << std::endl;
    std::cout << "Expected Function Value: ";
    std::cout << std::fixed << std::setprecision(12) << expectedFx << std::endl;
    std::cout << "Theta Gradient: ";
    std::cout << thetaGrad.transpose().format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta Gradient: ";
    std::cout << expectedThetaGrad.transpose().format(HeavyFormat) << std::endl;

    ASSERT_TRUE(abs(fx - expectedFx) <= 1e-6);
    ASSERT_TRUE(thetaGrad.isApprox(expectedThetaGrad));
}

TEST(MC3DModel, AddFrameOverBoundaries)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::vector<int> KEYPOINTS = {5};
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    double keypointConfidenceThreshold = 0.5;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;
    gmmParam.maxFrameBuffer = 2;
    gmmParam.autoManageTheta = true;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    // ASSERT_EQ(gmm.designMatrix.rows(), 0);
    // ASSERT_EQ(gmm.designMatrix.cols(), 0);
    ASSERT_EQ(gmm.spline.getNumBasis(), 0);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 0);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 0);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    std::cout << "Adding Frame 0" << std::endl;

    gmm.addFrame(frames[0]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 1);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.spline.getNumBasis(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 1);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    std::cout << "Adding Frame 1" << std::endl;

    gmm.addFrame(frames[1]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 2);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.spline.getNumBasis(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 2);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    // std::cout << gmm.gmmContainers[5].m_keypoints << std::endl;

    std::cout << "Adding Frame 2" << std::endl;

    gmm.addFrame(frames[2]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 2);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.spline.getNumBasis(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 2);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    // std::cout << gmm.gmmContainers[5].m_keypoints << std::endl;

    gmm.fit();
}

TEST(MC3DModel, AddFrameOverBoundaries2)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::vector<int> KEYPOINTS = {5};
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 1;
    gmmParam.maxFrameBuffer = 3;
    gmmParam.autoManageTheta = true;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    // ASSERT_EQ(gmm.designMatrix.rows(), 0);
    // ASSERT_EQ(gmm.designMatrix.cols(), 0);
    ASSERT_EQ(gmm.spline.getNumBasis(), 0);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 0);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 0);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    std::cout << "Adding Frame 0" << std::endl;

    gmm.addFrame(frames[0]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 1);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.spline.getNumBasis(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 1);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    std::cout << "Adding Frame 1" << std::endl;

    gmm.addFrame(frames[1]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 2);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.spline.getNumBasis(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 2);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    // std::cout << gmm.gmmContainers[5].m_keypoints << std::endl;

    std::cout << "Adding Frame 2" << std::endl;

    gmm.addFrame(frames[2]);

    std::cout << "gmm.gmmContainers[5].parameters.theta" << std::endl;
    std::cout << gmm.gmmContainers[5].parameters.theta << std::endl;

    // ASSERT_EQ(gmm.designMatrix.rows(), 3);
    // ASSERT_EQ(gmm.designMatrix.cols(), 5);
    ASSERT_EQ(gmm.spline.getNumBasis(), 5);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 3);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[5].parameters.theta.cols(), 3);

    // std::cout << gmm.gmmContainers[5].m_keypoints << std::endl;

    gmm.fit();
}

TEST(MC3DModel, AddFrameInBoundariesMultiplePointsPerFrame)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<Frame<double>> frames = initAllFramesDouble();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::vector<int> KEYPOINTS = {5};
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    double keypointConfidenceThreshold = 0.5;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;
    gmmParam.maxFrameBuffer = 2;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;
    lbfgsParam.epsilon_rel = 1e-2;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    std::cout << "Adding Frame 0" << std::endl;

    gmm.addFrame(frames[0]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 1);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 2);

    gmm.prepareDesignMatrix();
    std::cout << "hGrads:" << std::endl;
    for (auto hGrad : gmm.hGrads)
    {
        std::cout << hGrad << std::endl;
    }

    std::cout << "Adding Frame 1" << std::endl;

    gmm.addFrame(frames[1]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 2);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 4);

    gmm.prepareDesignMatrix();
    std::cout << "hGrads:" << std::endl;
    for (auto hGrad : gmm.hGrads)
    {
        std::cout << hGrad << std::endl;
    }

    std::cout << "Adding Frame 2" << std::endl;

    gmm.addFrame(frames[2]);

    // ASSERT_EQ(gmm.designMatrix.rows(), 2);
    // ASSERT_EQ(gmm.designMatrix.cols(), 4);
    ASSERT_EQ(gmm.gmmContainers[5].keyPoints.size(), 4);

    gmm.prepareDesignMatrix();
    std::cout << "hGrads:" << std::endl;
    for (auto hGrad : gmm.hGrads)
    {
        std::cout << hGrad << std::endl;
    }

    auto res = gmm.fit();
}