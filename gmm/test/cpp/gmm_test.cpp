#include "mc3d_common.h"
#include "gmm.h"
#include "bspline.h"
#include "bspline.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "gmm_param.h"
#include "test_shared.h"
#include "test_double_shared.h"
#include "em.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iomanip>
#include <memory>

using namespace MC3D_TRECSIM;

TEST(GMM, Init)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};

    unsigned int J = 1;

    std::vector<RowMatrix<double>> frame1Kpts = { kptsFrame1Person0 };
    std::vector<Frame<double>> frames = { Frame<double>(0, frame1Kpts, 1726.080322265625, 1701415521134.211) };
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, 1725.5244140625, 1814.42578125);
    std::vector<int> KEYPOINTS = { 0, 5, 6 };
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    double nu = 1.0;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1e-1;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    for (auto frame: frames)
    {
        gmm.addFrame(frame);
    }

    gmm.prepareDesignMatrix();

    RowMatrix<double> expectedCovariance(2, 2);
    expectedCovariance << 1.0, 0.0, 0.0, 1.0;
    RowMatrix<double> expectedDesignMatrix(1, 4);
    expectedDesignMatrix << 0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0;
    gmm.spline.smoothDesignMatrix(expectedDesignMatrix, 0);

    std::cout << "Design Matrix:" << std::endl;
    std::cout << gmm.designMatrix.format(HeavyFormat) << std::endl;
    std::cout << "Expected Design Matrix:" << std::endl;
    std::cout << expectedDesignMatrix.format(HeavyFormat) << std::endl;

    ASSERT_EQ(gmm.gmmContainers.size(), 3);
    ASSERT_TRUE(gmm.gmmContainers[0].mvn.getCovariance().isApprox(expectedCovariance));
    ASSERT_TRUE(gmm.gmmContainers[5].mvn.getCovariance().isApprox(expectedCovariance));
    ASSERT_TRUE(gmm.gmmContainers[6].mvn.getCovariance().isApprox(expectedCovariance));
    ASSERT_TRUE(gmm.designMatrix.isApprox(expectedDesignMatrix));
}

TEST(GMM, AddMultipleFrames)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<RowMatrix<double>> frame1Kpts = {kptsFrame1Person0};
    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, 1725.5244140625, 1814.42578125);
    std::vector<int> KEYPOINTS = {0, 5, 6};
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    double nu = 1.0;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1e-1;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;
    gmmParam.nu = 12;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    gmm.prepareCalculations();

    RowMatrix<double> expectedCovariance(2, 2);
    expectedCovariance << 12.0, 0.0, 0.0, 12.0;
    RowMatrix<double> expectedDesignMatrix(10, 5);
    expectedDesignMatrix << 0.166666666666667, 0.666666666666667, 0.166666666666667, 0, 0, 0.160565899317987, 0.666514999949354, 0.172918786522331, 3.14210328118255e-07, 0, 0.0211571264607653, 0.480775647782328, 0.477554364983603, 0.0205128607733032, 0, 0.0200717297056553, 0.475301883693101, 0.483012423705165, 0.0216139628960784, 0, 4.23769039456627e-09, 0.168141246433254, 0.666658032614865, 0.165200716714191, 0, 0, 0.164597382121371, 0.666649431747418, 0.168753174168829, 1.19623821782341e-08, 0, 0.023274833507054, 0.490834274384573, 0.46732205487636, 0.0185688372320139, 0, 0.0213847314121679, 0.481894863111771, 0.476428909401065, 0.0202914960749959, 0, 7.15514184280235e-06, 0.18476582852666, 0.665462068493374, 0.149764947838123, 0, 2.42531920472781e-06, 0.179164445182929, 0.666077896176527, 0.154755233321339;
    gmm.spline.smoothDesignMatrix(expectedDesignMatrix, 0);

    std::cout << "Design Matrix:" << std::endl;
    std::cout << gmm.designMatrix.format(HeavyFormat) << std::endl;
    std::cout << "Expected Design Matrix:" << std::endl;
    std::cout << expectedDesignMatrix.format(HeavyFormat) << std::endl;
    std::cout << "Covariance 0:" << std::endl;
    std::cout << gmm.gmmContainers[0].mvn.getCovariance().format(HeavyFormat) << std::endl;
    std::cout << "Covariance 5:" << std::endl;
    std::cout << gmm.gmmContainers[5].mvn.getCovariance().format(HeavyFormat) << std::endl;
    std::cout << "Covariance 6:" << std::endl;
    std::cout << gmm.gmmContainers[6].mvn.getCovariance().format(HeavyFormat) << std::endl;
    std::cout << "Expected Covariance:" << std::endl;
    std::cout << expectedCovariance.format(HeavyFormat) << std::endl;

    ASSERT_EQ(gmm.gmmContainers.size(), 3);
    ASSERT_TRUE(gmm.gmmContainers[0].mvn.getCovariance().isApprox(expectedCovariance));
    ASSERT_TRUE(gmm.gmmContainers[5].mvn.getCovariance().isApprox(expectedCovariance));
    ASSERT_TRUE(gmm.gmmContainers[6].mvn.getCovariance().isApprox(expectedCovariance));
    ASSERT_TRUE(gmm.designMatrix.isApprox(expectedDesignMatrix));
}

TEST(GMM, H)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size()-1).time);
    std::vector<int> KEYPOINTS = { 0, 5, 6 };
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    double nu = 1.0;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1e-1;

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
    gmm.gmmContainers[0].parameters.theta = theta.transpose();

    ColMatrix<double> h = gmm.h(0, gmm.gmmContainers[0].parameters.theta);
    ColMatrix<double> expectedH(3, 1);
    expectedH << 3.182318810497387318e+01, -2.909158142026773675e+01, 2.887777572907002011e+01;

    ASSERT_TRUE(h.isApprox(expectedH));
}

TEST(GMM, Fit)
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
    double nu = 1.0;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1.0;
    int keypoint = 5;

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

    ColMatrix<double> theta(3, 5);
    theta << -1.254598811526375002e+01, 4.507143064099161478e+01, 2.319939418114050866e+01, 9.865848419703659999e+00, -3.439813595575634508e+01, -3.440054796637973311e+01, -4.419163878318005345e+01, 3.661761457749351933e+01, 1.011150117432088003e+01, 2.080725777960454792e+01, -4.794155057041975709e+01, 4.699098521619943369e+01, 3.324426408004217137e+01, -2.876608893217238361e+01, -3.181750327928993727e+01;
    theta = theta.transpose().eval();
    //gmm.gmmContainers[keypoint].m_initialTheta = theta;
    
    auto tstart = std::chrono::steady_clock::now();
    std::map<int, EMFitResult<double>> fitResults = gmm.fit();
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "Time Neede: ";
    std::cout << cputime.count();
    std::cout << "s" << std::endl;

    ColMatrix<double> expectedTheta(5, 3);
    expectedTheta << 73.2623623740736, -7.59876032436745,  225.234325008981, 52.3161618462083, -43.7879074790608, 273.40614617857 , 56.7366404709181, -59.4195441713384, 288.303891802536 , 42.6942791376544, -58.3868518751559, 264.809577443842, 71.8579794155048, -134.442621887032,  477.290284887415;
    EMFitResult<double> fitResult = fitResults[keypoint];

    std::cout << "Initial Theta:" << std::endl;
    std::cout << theta.format(HeavyFormat) << std::endl;
    std::cout << "Responsibilites:" << std::endl;
    std::cout << fitResult.responsibilities << std::endl;
    std::cout << "Found Theta:" << std::endl;
    std::cout << fitResult.parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta:" << std::endl;
    std::cout << expectedTheta.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 1);
    ASSERT_TRUE(fitResult.parameters.theta.isApprox(expectedTheta, 1e-1));
}


TEST(GMM, FitAfterMultipleFrameAdds)
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
    double nu = 1.0;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1.0;
    int keypoint = 5;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);
    std::map<int, EMFitResult<double>> fitResults;

    for (int i = 0; i < 5; i++)
    {
        gmm.addFrame(frames[i]);
    }
    fitResults = gmm.fit();

    for (int i = 5; i < frames.size(); i++)
    {
        gmm.addFrame(frames[i]);
        fitResults = gmm.fit();
        std::cout << "Found Theta:" << std::endl;
        EMFitResult<double> fitResult = fitResults[keypoint];
        std::cout << fitResult.parameters.theta.format(HeavyFormat) << std::endl;
    }

    ColMatrix<double> expectedTheta(5, 3);
    expectedTheta << 73.2623623740736, -7.59876032436745, 225.234325008981, 52.3161618462083, -43.7879074790608, 273.40614617857, 56.7366404709181, -59.4195441713384, 288.303891802536, 42.6942791376544, -58.3868518751559, 264.809577443842, 71.8579794155048, -134.442621887032, 477.290284887415;
    EMFitResult<double> fitResult = fitResults[keypoint];

    std::cout << "Responsibilites:" << std::endl;
    std::cout << fitResult.responsibilities << std::endl;
    std::cout << "Found Theta:" << std::endl;
    std::cout << fitResult.parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta:" << std::endl;
    std::cout << expectedTheta.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 1);
    ASSERT_TRUE(fitResult.parameters.theta.isApprox(expectedTheta, 1e-1));
}

TEST(GMM, CheckAutoThetaIncrease)
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
    double nu = 1.0;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1.0;
    int keypoint = 5;

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

    ColMatrix<double> theta(3, 5);
    theta << -1.254598811526375002e+01, 4.507143064099161478e+01, 2.319939418114050866e+01, 9.865848419703659999e+00, -3.439813595575634508e+01, -3.440054796637973311e+01, -4.419163878318005345e+01, 3.661761457749351933e+01, 1.011150117432088003e+01, 2.080725777960454792e+01, -4.794155057041975709e+01, 4.699098521619943369e+01, 3.324426408004217137e+01, -2.876608893217238361e+01, -3.181750327928993727e+01;
    theta = theta.transpose().eval();
    // gmm.gmmContainers[keypoint].m_initialTheta = theta;

    auto tstart = std::chrono::steady_clock::now();
    std::map<int, EMFitResult<double>> fitResults = gmm.fit();
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "Time Neede: ";
    std::cout << cputime.count();
    std::cout << "s" << std::endl;

    ColMatrix<double> expectedTheta(5, 3);
    expectedTheta << 73.2623623740736, -7.59876032436745, 225.234325008981, 52.3161618462083, -43.7879074790608, 273.40614617857, 56.7366404709181, -59.4195441713384, 288.303891802536, 42.6942791376544, -58.3868518751559, 264.809577443842, 71.8579794155048, -134.442621887032, 477.290284887415;
    EMFitResult<double> fitResult = fitResults[keypoint];

    std::cout << "Initial Theta:" << std::endl;
    std::cout << theta.format(HeavyFormat) << std::endl;
    std::cout << "Responsibilites:" << std::endl;
    std::cout << fitResult.responsibilities << std::endl;
    std::cout << "Found Theta:" << std::endl;
    std::cout << fitResult.parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta:" << std::endl;
    std::cout << expectedTheta.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 1);
    ASSERT_TRUE(fitResult.parameters.theta.isApprox(expectedTheta, 1e-1));
}

TEST(GMM, FitTwoPeopleWithOneHypothesis)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<Frame<double>> frames = initAllFramesDouble(500);
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::vector<int> KEYPOINTS = {5};
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1.0;
    int keypoint = 5;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;
    gmmParam.nu = 1.0;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    ColMatrix<double> theta(3, 5);
    theta << -1.254598811526375002e+01, 4.507143064099161478e+01, 2.319939418114050866e+01, 9.865848419703659999e+00, -3.439813595575634508e+01, -3.440054796637973311e+01, -4.419163878318005345e+01, 3.661761457749351933e+01, 1.011150117432088003e+01, 2.080725777960454792e+01, -4.794155057041975709e+01, 4.699098521619943369e+01, 3.324426408004217137e+01, -2.876608893217238361e+01, -3.181750327928993727e+01;
    theta = theta.transpose().eval();
    //gmm.gmmContainers[keypoint].m_initialTheta = theta;

    auto tstart = std::chrono::steady_clock::now();
    std::map<int, EMFitResult<double>> fitResults = gmm.fit();
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "Time Neede: ";
    std::cout << cputime.count();
    std::cout << "s" << std::endl;

    ColMatrix<double> expectedTheta(5, 3);
    expectedTheta << 88.132550956683,  71.190731209761, 182.485120538197, 92.539639962502,  25.955390960899, 232.714038825165, 97.202099778116,  23.375397392222, 246.133879718729, 82.187143518612,  13.888219773659, 225.68126472408, 154.463589750219,   6.691543910621, 431.746349374697;
    EMFitResult<double> fitResult = fitResults[keypoint];

    std::cout << "Initial Theta:" << std::endl;
    std::cout << theta.format(HeavyFormat) << std::endl;
    std::cout << "Responsibilites:" << std::endl;
    std::cout << fitResult.responsibilities << std::endl;
    std::cout << "Found Theta:" << std::endl;
    std::cout << fitResult.parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta:" << std::endl;
    std::cout << expectedTheta.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 1);
    ASSERT_TRUE(fitResult.parameters.theta.isApprox(expectedTheta, 1e-1));
}


TEST(GMM, FitTwoPeopleWithTwoHypothesisWithPredefinedThetas)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 2;

    std::vector<Frame<double>> frames = initAllDoubleFrames();
    std::vector<int> KEYPOINTS = {5};
    std::map<int, Vector<double>> initialPis;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.tol = 1e-6;
    gmmParam.splineKnotDelta = 200;
    gmmParam.maxFrameBuffer = 20;
    gmmParam.nu = 500;
    gmmParam.autoManageTheta = true;
    gmmParam.copyLastThetas = true;
    gmmParam.splineSmoothingFactor = 10;
    gmmParam.maxIter = 5;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 40;
    lbfgsParam.max_linesearch = 10;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    ColMatrix<double> theta(6, 4);
    theta << 11.657898127378987, 22.25710602936815, 41.83582507131999, 95.63691038207142,
        15.901146752736414, -6.975190610727871, -25.249031854647892, -165.62066827434148,
        43.6778724999175, 76.27704028025457, 197.08503221745778, 361.2090380471671,
        -122.38815780512078, -91.77560416107559, -98.13081179207634, -183.32971238335895,
        -296.5684027594996, -284.75694155626553, -281.08872608955056, -254.04764750341602,
        582.5363181945265, 537.2634422889759, 540.4704728037173, 471.0919214491662;
    theta = theta.transpose().eval();
    std::map<int, ColMatrix<double>> initialThetas{{5, theta}};

    std::cout << "Initial Theta:" << std::endl;
    std::cout << theta.format(HeavyFormat) << std::endl;

    auto tstart = std::chrono::steady_clock::now();
    std::map<int, EMFitResult<double>> fitResults = gmm.fit(initialThetas);
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "Time Neede: ";
    std::cout << cputime.count();
    std::cout << "s" << std::endl;

    EMFitResult<double> fitResult = fitResults[5];

    std::cout << "Found Theta:" << std::endl;
    std::cout << fitResult.parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Found Pi:" << std::endl;
    std::cout << fitResult.parameters.pi.format(HeavyFormat) << std::endl;
    std::cout << "Convergence:" << std::endl;
    std::cout << fitResult.convergence << std::endl;
    std::cout << "Diff Theta:" << std::endl;
    std::cout << fitResult.diff << std::endl;
    std::cout << "Responsibilites:" << std::endl;
    std::cout << fitResult.responsibilities << std::endl;
    std::cout << "Expected Theta:" << std::endl;
    std::cout << theta.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 1);
    ASSERT_TRUE(fitResult.parameters.theta.isApprox(theta, 1e-1));
}

TEST(GMM, FitTwoKeypoints)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;
    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;

    int rightShoulderKeypoint = 5;
    int rightElbowKeypoint = 7;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = {5, 7};
    gmmParam.splineKnotDelta = 45;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(J, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    auto tstart = std::chrono::steady_clock::now();
    std::map<int, EMFitResult<double>> fitResults = gmm.fit();
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "Time Neede: ";
    std::cout << cputime.count();
    std::cout << "s" << std::endl;

    ColMatrix<double> expectedThetaRightShoulder(5, 3);
    expectedThetaRightShoulder << 73.2815459104087, -8.32126984742543,  226.689584736263, 52.308848686218, -43.6237316103347, 272.998359575625, 56.7444512194102, -59.3529933717434, 288.467476458373, 43.0288027294688, -58.2739238259099, 264.880363765974, 69.4920496751196, -129.596978535582,  463.419030976172;
    ColMatrix<double> expectedThetaRightElbow(5, 3);
    expectedThetaRightElbow << 62.2281725794883, -121.369848805468, 62.007396274245, 26.0228047424397, -23.4755245823039,  305.731200043973, 32.0388077570727, -85.0877406051797,  268.576165651193, 25.9487519142245, -59.6209019541959,  276.134963677695, 18.1528891576515, -132.947650789204,  394.637961387223;

    std::cout << "Found Theta (Right Shoulder):" << std::endl;
    std::cout << fitResults[rightShoulderKeypoint].parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta (Right Shoulder):" << std::endl;
    std::cout << expectedThetaRightShoulder.format(HeavyFormat) << std::endl;

    std::cout << "Found Theta (Right Elbow):" << std::endl;
    std::cout << fitResults[rightElbowKeypoint].parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta (Right Elbow):" << std::endl;
    std::cout << expectedThetaRightElbow.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 2);
    ASSERT_TRUE(fitResults[rightShoulderKeypoint].parameters.theta.isApprox(expectedThetaRightShoulder, 1e-1));
    ASSERT_TRUE(fitResults[rightElbowKeypoint].parameters.theta.isApprox(expectedThetaRightElbow, 1e-1));
}

TEST(GMM, AddHypothesis)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;

    int rightShoulderKeypoint = 5;
    int rightElbowKeypoint = 7;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = {5, 7};
    gmmParam.splineKnotDelta = 45;
    gmmParam.autoManageTheta = true;
    gmmParam.autoManageHypothesis = false;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(0, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }
    
    gmm.addHypothesis();

    ASSERT_EQ(gmm.gmmContainers.size(), 2);

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 3);

    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 3);

    gmm.addHypothesis();

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 6);

    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 6);

    gmm.addHypothesis();

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 9);

    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 9);

    gmm.addHypothesis();

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 12);

    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 12);

    gmm.addHypothesis();

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 15);

    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 15);
}

TEST(GMM, RemoveHypothesis)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;

    int rightShoulderKeypoint = 5;
    int rightElbowKeypoint = 7;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = {5, 7};
    gmmParam.splineKnotDelta = 45;
    gmmParam.autoManageTheta = true;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(0, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    gmm.addHypothesis();
    gmm.addHypothesis();
    gmm.addHypothesis();
    gmm.addHypothesis();
    gmm.addHypothesis();

    ColMatrix<double> oldThetaRightShoulder = gmm.gmmContainers[rightShoulderKeypoint].parameters.theta;
    ColMatrix<double> oldThetaRightElbow = gmm.gmmContainers[rightElbowKeypoint].parameters.theta;

    gmm.removeHypothesis(0);

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols() ,12);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 12);
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.isApprox(oldThetaRightShoulder.rightCols(12)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.isApprox(oldThetaRightElbow.rightCols(12)));

    gmm.addHypothesis();

    oldThetaRightShoulder = gmm.gmmContainers[rightShoulderKeypoint].parameters.theta;
    oldThetaRightElbow = gmm.gmmContainers[rightElbowKeypoint].parameters.theta;

    gmm.removeHypothesis(1);

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 12);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 12);
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.leftCols(3).isApprox(oldThetaRightShoulder.leftCols(3)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.leftCols(3).isApprox(oldThetaRightElbow.leftCols(3)));
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rightCols(9).isApprox(oldThetaRightShoulder.rightCols(9)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rightCols(9).isApprox(oldThetaRightElbow.rightCols(9)));

    gmm.addHypothesis();

    oldThetaRightShoulder = gmm.gmmContainers[rightShoulderKeypoint].parameters.theta;
    oldThetaRightElbow = gmm.gmmContainers[rightElbowKeypoint].parameters.theta;

    gmm.removeHypothesis(2);

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 12);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 12);
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.leftCols(6).isApprox(oldThetaRightShoulder.leftCols(6)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.leftCols(6).isApprox(oldThetaRightElbow.leftCols(6)));
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rightCols(6).isApprox(oldThetaRightShoulder.rightCols(6)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rightCols(6).isApprox(oldThetaRightElbow.rightCols(6)));

    gmm.addHypothesis();

    oldThetaRightShoulder = gmm.gmmContainers[rightShoulderKeypoint].parameters.theta;
    oldThetaRightElbow = gmm.gmmContainers[rightElbowKeypoint].parameters.theta;

    gmm.removeHypothesis(3);

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 12);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 12);
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.leftCols(9).isApprox(oldThetaRightShoulder.leftCols(9)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.leftCols(9).isApprox(oldThetaRightElbow.leftCols(9)));
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rightCols(3).isApprox(oldThetaRightShoulder.rightCols(3)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rightCols(3).isApprox(oldThetaRightElbow.rightCols(3)));

    gmm.addHypothesis();

    oldThetaRightShoulder = gmm.gmmContainers[rightShoulderKeypoint].parameters.theta;
    oldThetaRightElbow = gmm.gmmContainers[rightElbowKeypoint].parameters.theta;

    gmm.removeHypothesis(4);

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 12);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 12);
    ASSERT_TRUE(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.isApprox(oldThetaRightShoulder.leftCols(12)));
    ASSERT_TRUE(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.isApprox(oldThetaRightElbow.leftCols(12)));
}



TEST(GMM, FitWithoutHypothesis)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;

    int rightShoulderKeypoint = 5;
    int rightElbowKeypoint = 7;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = {5, 7};
    gmmParam.splineKnotDelta = 45;
    gmmParam.autoManageTheta = true;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(0, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    ASSERT_EQ(gmm.gmmContainers.size(), 2);

    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightShoulderKeypoint].parameters.theta.cols(), 0);

    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.rows(), 5);
    ASSERT_EQ(gmm.gmmContainers[rightElbowKeypoint].parameters.theta.cols(), 0);

    std::map<int, EMFitResult<double>> fitResults = gmm.fit();

    ASSERT_EQ(fitResults.size(), 2);

    for(auto fitResult : fitResults)
    {
        ASSERT_EQ(fitResult.second.parameters.theta.rows(), 5);
        ASSERT_EQ(fitResult.second.parameters.theta.cols(), 0);
        ASSERT_EQ(fitResult.second.parameters.pi.size(), 0);
        ASSERT_DOUBLE_EQ(fitResult.second.diff, std::numeric_limits<double>::infinity());
        ASSERT_EQ(fitResult.second.convergence, false);
    }
}

TEST(GMM, FitTwoKeypointsAfterAddHypothesis)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;

    int rightShoulderKeypoint = 5;
    int rightElbowKeypoint = 7;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = {5, 7};
    gmmParam.splineKnotDelta = 45;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(0, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    gmm.addHypothesis();

    std::cout << "size: " << gmm.gmmContainers[5].supports.size() << std::endl;

    auto tstart = std::chrono::steady_clock::now();
    std::map<int, EMFitResult<double>> fitResults = gmm.fit();
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "Time Needed: ";
    std::cout << cputime.count();
    std::cout << "s" << std::endl;

    ColMatrix<double> expectedThetaRightShoulder(5, 3);
    expectedThetaRightShoulder << 73.2815459104087, -8.32126984742543, 226.689584736263, 52.308848686218, -43.6237316103347, 272.998359575625, 56.7444512194102, -59.3529933717434, 288.467476458373, 43.0288027294688, -58.2739238259099, 264.880363765974, 69.4920496751196, -129.596978535582, 463.419030976172;
    ColMatrix<double> expectedThetaRightElbow(5, 3);
    expectedThetaRightElbow << 62.2281725794883, -121.369848805468, 62.007396274245, 26.0228047424397, -23.4755245823039, 305.731200043973, 32.0388077570727, -85.0877406051797, 268.576165651193, 25.9487519142245, -59.6209019541959, 276.134963677695, 18.1528891576515, -132.947650789204, 394.637961387223;

    std::cout << "Found Theta (Right Shoulder):" << std::endl;
    std::cout << fitResults[rightShoulderKeypoint].parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta (Right Shoulder):" << std::endl;
    std::cout << expectedThetaRightShoulder.format(HeavyFormat) << std::endl;

    std::cout << "Found Theta (Right Elbow):" << std::endl;
    std::cout << fitResults[rightElbowKeypoint].parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta (Right Elbow):" << std::endl;
    std::cout << expectedThetaRightElbow.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 2);
    ASSERT_TRUE(fitResults[rightShoulderKeypoint].parameters.theta.isApprox(expectedThetaRightShoulder, 1e-1));
    ASSERT_TRUE(fitResults[rightElbowKeypoint].parameters.theta.isApprox(expectedThetaRightElbow, 1e-1));
}

TEST(GMM, FitTwoKeypointsAfterAddAndRemoveHypothesis)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;

    int rightShoulderKeypoint = 5;
    int rightElbowKeypoint = 7;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = {5, 7};
    gmmParam.splineKnotDelta = 45;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    GMM<double> gmm(0, cameras, gmmParam, lbfgsParam);

    for (auto frame : frames)
    {
        gmm.addFrame(frame);
    }

    gmm.addHypothesis();
    gmm.addHypothesis();
    gmm.removeHypothesis(0);

    auto tstart = std::chrono::steady_clock::now();
    std::map<int, EMFitResult<double>> fitResults = gmm.fit();
    auto tend = std::chrono::steady_clock::now();
    std::chrono::duration<double> cputime = tend - tstart;
    std::cout << "Time Neede: ";
    std::cout << cputime.count();
    std::cout << "s" << std::endl;

    ColMatrix<double> expectedThetaRightShoulder(5, 3);
    expectedThetaRightShoulder << 73.2815459104087, -8.32126984742543, 226.689584736263, 52.308848686218, -43.6237316103347, 272.998359575625, 56.7444512194102, -59.3529933717434, 288.467476458373, 43.0288027294688, -58.2739238259099, 264.880363765974, 69.4920496751196, -129.596978535582, 463.419030976172;
    ColMatrix<double> expectedThetaRightElbow(5, 3);
    expectedThetaRightElbow << 62.2281725794883, -121.369848805468, 62.007396274245, 26.0228047424397, -23.4755245823039, 305.731200043973, 32.0388077570727, -85.0877406051797, 268.576165651193, 25.9487519142245, -59.6209019541959, 276.134963677695, 18.1528891576515, -132.947650789204, 394.637961387223;

    std::cout << "Found Theta (Right Shoulder):" << std::endl;
    std::cout << fitResults[rightShoulderKeypoint].parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta (Right Shoulder):" << std::endl;
    std::cout << expectedThetaRightShoulder.format(HeavyFormat) << std::endl;

    std::cout << "Found Theta (Right Elbow):" << std::endl;
    std::cout << fitResults[rightElbowKeypoint].parameters.theta.format(HeavyFormat) << std::endl;
    std::cout << "Expected Theta (Right Elbow):" << std::endl;
    std::cout << expectedThetaRightElbow.format(HeavyFormat) << std::endl;

    ASSERT_EQ(fitResults.size(), 2);
    ASSERT_TRUE(fitResults[rightShoulderKeypoint].parameters.theta.isApprox(expectedThetaRightShoulder, 1e-1));
    ASSERT_TRUE(fitResults[rightElbowKeypoint].parameters.theta.isApprox(expectedThetaRightElbow, 1e-1));
}

GMM<double> initStandardGMM()
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    unsigned int J = 1;

    std::vector<Frame<double>> frames = initAllFrames();
    Vector<double> splineKnots = Vector<double>::LinSpaced(3, frames.at(0).time, frames.at(frames.size() - 1).time);
    std::vector<int> KEYPOINTS = {0, 5, 6};
    std::map<int, ColMatrix<double>> initialThetas;
    std::map<int, Vector<double>> initialPis;
    double nu = 1.0;
    unsigned int maxIter = 100;
    unsigned int maxMinimizeIter = 100;
    double minimizeEpsilon = 1e-6;
    double keypointConfidenceThreshold = 0.5;
    double tol = 1e-1;

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = KEYPOINTS;
    gmmParam.splineKnotDelta = 45;

    LBFGSpp::LBFGSParam<double> lbfgsParam;
    lbfgsParam.max_iterations = 100;
    lbfgsParam.epsilon = 1e-6;

    return GMM<double>(J, cameras, gmmParam, lbfgsParam);
}