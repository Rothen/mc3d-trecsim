#include "mc3d_common.h"
#include "gmm.h"
#include "bspline.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "gmm_param.h"
#include "gmm_container.h"
#include "test_shared.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iomanip>
#include <memory>
#include <cstdarg>
#include <LBFGS.h>

using namespace MC3D_TRECSIM;

double calcFiniteDiff(const std::function<double(double)> &fn, double h = 1e-8)
{
    const double x1 = fn(h);
    const double x2 = fn(-h);

    return (x1 - x2) / (2 * h);
}

template <typename T>
T calcFiniteDiffMatrix(const std::function<double(T &)> &fn, T &value, double h = 1e-8)
{
    T finiteDiff = T::Zero(value.rows(), value.cols());

    for (int r = 0; r < value.rows(); r++)
    {
        for (int c = 0; c < value.cols(); c++)
        {
            const double oldvalue = value(r, c);
            value(r, c) += h;
            const double x1 = fn(value);
            value(r, c) = oldvalue - h;
            const double x2 = fn(value);
            value(r, c) = oldvalue;

            finiteDiff(r, c) = (x1 - x2) / (2 * h);
        }
    }

    return finiteDiff;
}

TEST(Derivatives, WholeModelFit)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};

    BSpline<double> bspline = BSpline<double>(3, AUGMENTATION_MODE::UNIFORM, Eigen::VectorXd::LinSpaced(3, 0, 1));
    RowMatrix<double> responsibilities = RowMatrix<double>::Ones(4, 1);
    Vector<double> pi = Vector<double>::Ones(1);
    int keypoint = 5;
    int J = 1;
    GMMParam<double> gmmParam = GMMParam<double>();
    gmmParam.splineSmoothingFactor = 10;

    RowMatrix<double> designMatrix = bspline.designMatrix(Eigen::VectorXd::LinSpaced(10, 0, 1));
    bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor);

    std::vector<RowMatrix<double>> hGrads;
    int numBasis = bspline.getNumBasis();

    for (int r = 0; r < designMatrix.rows(); r++)
    {
        Vector<double> designMatrixRow = designMatrix.row(r);
        RowMatrix<double> hGrad = RowMatrix<double>::Zero(3, numBasis * 3);
        hGrad.row(0).head(numBasis) = designMatrixRow;
        hGrad.row(1).segment(numBasis, numBasis) = designMatrixRow;
        hGrad.row(2).tail(numBasis) = designMatrixRow;
        hGrads.push_back(hGrad);
    }

    double nu = 1e-2;
    RowMatrix<double> thetaLimiter = RowMatrix<double>::Ones(numBasis, J * 3) * 2;

    GMMContainer<double> gmmContainer = GMMContainer<double>(keypoint, J, cameras, nu, designMatrix);
    gmmContainer.parameters.pi = pi;
    gmmContainer.addKeypoint(Vector<double>::Random(2), 0.5, 0, 0);
    gmmContainer.addKeypoint(Vector<double>::Random(2), 0.6, 1, 1);
    gmmContainer.addKeypoint(Vector<double>::Random(2), 1.1, 0, 2);
    gmmContainer.addKeypoint(Vector<double>::Random(2), 1.3, 1, 3);

    MC3DModel<double> mc3dModel = MC3DModel<double>(bspline, responsibilities, gmmContainer, hGrads, cameras);

    for (int i = 0; i < 1000; i++)
    {
        RowMatrix<double> theta = (RowMatrix<double>::Random(numBasis, J * 3) + thetaLimiter) * 50;

        std::function<double(RowMatrix<double> &)> fn([&](RowMatrix<double> &newTheta) -> double
        {
            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            return mc3dModel(newTheta.reshaped().transpose(), thetaGrad);
        });

        RowMatrix<double> finiteDiff = calcFiniteDiffMatrix(fn, theta, 1e-4);

        Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
        Vector<double> thetaVec = theta.reshaped().transpose();
        mc3dModel(thetaVec, thetaGrad);
        RowMatrix<double> analyticDiff = thetaGrad.reshaped(numBasis, J * 3);

        double isApprox = analyticDiff.isApprox(finiteDiff, 1e-2);
        
        if (!isApprox)
        {
            std::cout << "Analytic: " << std::endl << analyticDiff.format(HeavyFormat) << std::endl;
            std::cout << "Finite: " << std::endl << finiteDiff.format(HeavyFormat) << std::endl;
        }

        ASSERT_TRUE(isApprox);
    }
}

TEST(Derivatives, PointsOnlyModelFit)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};

    BSpline<double> bspline = BSpline<double>(3, AUGMENTATION_MODE::UNIFORM, Eigen::VectorXd::LinSpaced(3, 0, 1));
    RowMatrix<double> responsibilities = RowMatrix<double>::Ones(4, 1);
    Vector<double> pi = Vector<double>::Ones(1);
    int keypoint = 5;
    int J = 1;
    GMMParam<double> gmmParam = GMMParam<double>();
    gmmParam.splineSmoothingFactor = 0;

    RowMatrix<double> designMatrix = bspline.designMatrix(Eigen::VectorXd::LinSpaced(10, 0, 1));
    bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor);

    std::vector<RowMatrix<double>> hGrads;
    int numBasis = bspline.getNumBasis();

    for (int r = 0; r < designMatrix.rows(); r++)
    {
        Vector<double> designMatrixRow = designMatrix.row(r);
        RowMatrix<double> hGrad = RowMatrix<double>::Zero(3, numBasis * 3);
        hGrad.row(0).head(numBasis) = designMatrixRow;
        hGrad.row(1).segment(numBasis, numBasis) = designMatrixRow;
        hGrad.row(2).tail(numBasis) = designMatrixRow;
        hGrads.push_back(hGrad);
    }

    double nu = 1e-2;
    RowMatrix<double> thetaLimiter = RowMatrix<double>::Ones(numBasis, J * 3) * 2;

    GMMContainer<double> gmmContainer = GMMContainer<double>(keypoint, J, cameras, nu, designMatrix);
    gmmContainer.parameters.pi = pi;
    gmmContainer.addKeypoint(Vector<double>::Random(2), 0.5, 0, 0);
    gmmContainer.addKeypoint(Vector<double>::Random(2), 0.6, 1, 1);
    gmmContainer.addKeypoint(Vector<double>::Random(2), 1.1, 0, 2);
    gmmContainer.addKeypoint(Vector<double>::Random(2), 1.3, 1, 3);

    MC3DModel<double> mc3dModel = MC3DModel<double>(bspline, responsibilities, gmmContainer, hGrads, cameras);

    for (int i = 0; i < 1000; i++)
    {
        RowMatrix<double> theta = (RowMatrix<double>::Random(numBasis, J * 3) + thetaLimiter) * 50;

        std::function<double(RowMatrix<double> &)> fn([&](RowMatrix<double> &newTheta) -> double
        {
            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            return mc3dModel(newTheta.reshaped().transpose(), thetaGrad);
        });

        RowMatrix<double> finiteDiff = calcFiniteDiffMatrix(fn, theta, 1e-4);

        Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
        Vector<double> thetaVec = theta.reshaped().transpose();
        mc3dModel(thetaVec, thetaGrad);
        RowMatrix<double> analyticDiff = thetaGrad.reshaped(numBasis, J * 3);

        double isApprox = analyticDiff.isApprox(finiteDiff, 1e-2);
        
        if (!isApprox)
        {
            std::cout << "Analytic: " << std::endl << analyticDiff.format(HeavyFormat) << std::endl;
            std::cout << "Finite: " << std::endl << finiteDiff.format(HeavyFormat) << std::endl;
        }

        ASSERT_TRUE(isApprox);
    }
}

TEST(Derivatives, SmoothnessOnlyModelFit)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    BSpline<double> bspline = BSpline<double>(3, AUGMENTATION_MODE::UNIFORM, Eigen::VectorXd::LinSpaced(3, 0, 1));
    RowMatrix<double> responsibilities = RowMatrix<double>::Ones(4, 1);
    Vector<double> pi = Vector<double>::Ones(1);
    int keypoint = 5;
    int J = 1;
    GMMParam<double> gmmParam = GMMParam<double>();
    gmmParam.splineSmoothingFactor = 10;

    RowMatrix<double> designMatrix = bspline.designMatrix(Eigen::VectorXd::LinSpaced(10, 0, 1));
    bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor);

    std::vector<RowMatrix<double>> hGrads;
    int numBasis = bspline.getNumBasis();

    for (int r = 0; r < designMatrix.rows(); r++)
    {
        Vector<double> designMatrixRow = designMatrix.row(r);
        RowMatrix<double> hGrad = RowMatrix<double>::Zero(3, numBasis * 3);
        hGrad.row(0).head(numBasis) = designMatrixRow;
        hGrad.row(1).segment(numBasis, numBasis) = designMatrixRow;
        hGrad.row(2).tail(numBasis) = designMatrixRow;
        hGrads.push_back(hGrad);
    }

    double nu = 1e-2;

    GMMContainer<double> gmmContainer = GMMContainer<double>(keypoint, J, cameras, nu, designMatrix);
    gmmContainer.parameters.pi = pi;
    MC3DModel<double> mc3dModel = MC3DModel<double>(bspline, responsibilities, gmmContainer, hGrads, cameras);

    for (int i = 0; i < 1000; i++)
    {
        RowMatrix<double> theta = RowMatrix<double>::Random(numBasis, J * 3);

        std::function<double(RowMatrix<double> &)> fn([&](RowMatrix<double> &newTheta) -> double
        {
            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            return mc3dModel(newTheta.reshaped().transpose(), thetaGrad);
        });

        RowMatrix<double> finiteDiff = calcFiniteDiffMatrix(fn, theta, 1e-4);

        Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
        Vector<double> thetaVec = theta.reshaped().transpose();
        mc3dModel(thetaVec, thetaGrad);
        RowMatrix<double> analyticDiff = thetaGrad.reshaped(numBasis, J * 3);

        double isApprox = analyticDiff.isApprox(finiteDiff, 1e-6);

        if (!isApprox)
        {
            std::cout << "Analytic: " << std::endl << analyticDiff.format(HeavyFormat) << std::endl;
            std::cout << "Finite: " << std::endl << finiteDiff.format(HeavyFormat) << std::endl;
        }

        ASSERT_TRUE(isApprox);
    }
}

TEST(Derivatives, MultipleSmoothnessFactorsOnlyModelFit)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    BSpline<double> bspline = BSpline<double>(3, AUGMENTATION_MODE::UNIFORM, Eigen::VectorXd::LinSpaced(3, 0, 1));
    RowMatrix<double> responsibilities = RowMatrix<double>::Ones(4, 1);
    Vector<double> pi = Vector<double>::Ones(1);
    int keypoint = 5;
    int J = 1;
    double nu = 1e-2;
    Vector<double> splineSmoothingFactors = pow(10, Vector<double>::LinSpaced(13, -6, 6).array());

    for (int s = 0; s < splineSmoothingFactors.size(); s++)
    {
        GMMParam<double> gmmParam = GMMParam<double>();
        gmmParam.splineSmoothingFactor = splineSmoothingFactors[s];

        RowMatrix<double> designMatrix = bspline.designMatrix(Eigen::VectorXd::LinSpaced(10, 0, 1));
        bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor);

        std::vector<RowMatrix<double>> hGrads;
        int numBasis = bspline.getNumBasis();

        for (int r = 0; r < designMatrix.rows(); r++)
        {
            Vector<double> designMatrixRow = designMatrix.row(r);
            RowMatrix<double> hGrad = RowMatrix<double>::Zero(3, numBasis * 3);
            hGrad.row(0).head(numBasis) = designMatrixRow;
            hGrad.row(1).segment(numBasis, numBasis) = designMatrixRow;
            hGrad.row(2).tail(numBasis) = designMatrixRow;
            hGrads.push_back(hGrad);
        }

        GMMContainer<double> gmmContainer = GMMContainer<double>(keypoint, J, cameras, nu, designMatrix);
        gmmContainer.parameters.pi = pi;
        MC3DModel<double> mc3dModel = MC3DModel<double>(bspline, responsibilities, gmmContainer, hGrads, cameras);

        for (int i = 0; i < 1000; i++)
        {
            RowMatrix<double> theta = RowMatrix<double>::Random(numBasis, J * 3);

        std::function<double(RowMatrix<double> &)> fn([&](RowMatrix<double> &newTheta) -> double
        {
            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            return mc3dModel(newTheta.reshaped().transpose(), thetaGrad);
        });

        RowMatrix<double> finiteDiff = calcFiniteDiffMatrix(fn, theta, 1e-4);

            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            Vector<double> thetaVec = theta.reshaped().transpose();
            mc3dModel(thetaVec, thetaGrad);
            RowMatrix<double> analyticDiff = thetaGrad.reshaped(numBasis, J * 3);

            double isApprox = analyticDiff.isApprox(finiteDiff, 1e-6);

            if (!isApprox)
            {
                std::cout << "Spline Smoothing Factor: " << gmmParam.splineSmoothingFactor << std::endl;
                std::cout << "Analytic: " << std::endl << analyticDiff.format(HeavyFormat) << std::endl;
                std::cout << "Finite: " << std::endl << finiteDiff.format(HeavyFormat) << std::endl;
            }

            ASSERT_TRUE(isApprox);
        }
    }
}

TEST(Derivatives, SmoothnessPenalty)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    BSpline<double> bspline = BSpline<double>(3, AUGMENTATION_MODE::UNIFORM, Eigen::VectorXd::LinSpaced(3, 0, 1));
    RowMatrix<double> responsibilities = RowMatrix<double>::Zero(0, 0);
    Vector<double> pi = Vector<double>::Ones(1);
    int keypoint = 5;
    int J = 1;
    GMMParam<double> gmmParam = GMMParam<double>();
    gmmParam.splineSmoothingFactor = 10;

    RowMatrix<double> designMatrix = bspline.designMatrix(Eigen::VectorXd::LinSpaced(10, 0, 1));
    bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor);

    std::vector<RowMatrix<double>> hGrads;
    int numBasis = bspline.getNumBasis();

    for (int r = 0; r < designMatrix.rows(); r++)
    {
        Vector<double> designMatrixRow = designMatrix.row(r);
        RowMatrix<double> hGrad = RowMatrix<double>::Zero(3, numBasis * 3);
        hGrad.row(0).head(numBasis) = designMatrixRow;
        hGrad.row(1).segment(numBasis, numBasis) = designMatrixRow;
        hGrad.row(2).tail(numBasis) = designMatrixRow;
        hGrads.push_back(hGrad);
    }

    double nu = 1e-2;
    GMMContainer<double> gmmContainer = GMMContainer<double>(keypoint, J, cameras, nu, designMatrix);
    gmmContainer.parameters.pi = pi;
    MC3DModel<double> mc3dModel = MC3DModel<double>(bspline, responsibilities, gmmContainer, hGrads, cameras);

    for (int i = 0; i < 1000; i++)
    {
        RowMatrix<double> theta = RowMatrix<double>::Random(numBasis, J * 3);

        std::function<double(RowMatrix<double> &)> fn([&](RowMatrix<double> &newTheta) -> double
        {
            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            return mc3dModel(newTheta.reshaped().transpose(), thetaGrad);
        });

        RowMatrix<double> finiteDiff = calcFiniteDiffMatrix(fn, theta, 1e-4);

        double fx = 0;
        Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
        RowMatrix<double> PWs = designMatrix * theta;
        mc3dModel.addSmoothnessPenalties(fx, thetaGrad, numBasis, PWs);
        RowMatrix<double> analyticDiff = thetaGrad.reshaped(numBasis, J * 3);

        double isApprox = analyticDiff.isApprox(finiteDiff, 1e-6);
        
        if (!isApprox)
        {
            std::cout << "Analytic: " << std::endl << analyticDiff.format(HeavyFormat) << std::endl;
            std::cout << "Finite: " << std::endl << finiteDiff.format(HeavyFormat) << std::endl;
        }

        ASSERT_TRUE(isApprox);
    }
}

TEST(Derivatives, MultipleSmoothnessPenaltyFactors)
{
    Camera<double> camera0 = initCamera0();
    Camera<double> camera1 = initCamera1();
    std::vector<Camera<double>> cameras{camera0, camera1};
    BSpline<double> bspline = BSpline<double>(3, AUGMENTATION_MODE::UNIFORM, Eigen::VectorXd::LinSpaced(3, 0, 1));
    RowMatrix<double> responsibilities = RowMatrix<double>::Zero(0, 0);
    Vector<double> pi = Vector<double>::Ones(1);
    int keypoint = 5;
    int J = 1;
    double nu = 1e-2;
    Vector<double> splineSmoothingFactors = pow(10, Vector<double>::LinSpaced(13, -6, 6).array());

    for (int s = 0; s < splineSmoothingFactors.size(); s++)
    {
        GMMParam<double> gmmParam = GMMParam<double>();
        gmmParam.splineSmoothingFactor = splineSmoothingFactors[s];

        RowMatrix<double> designMatrix = bspline.designMatrix(Eigen::VectorXd::LinSpaced(10, 0, 1));
        bspline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor);

        std::vector<RowMatrix<double>> hGrads;
        int numBasis = bspline.getNumBasis();

        for (int r = 0; r < designMatrix.rows(); r++)
        {
            Vector<double> designMatrixRow = designMatrix.row(r);
            RowMatrix<double> hGrad = RowMatrix<double>::Zero(3, numBasis * 3);
            hGrad.row(0).head(numBasis) = designMatrixRow;
            hGrad.row(1).segment(numBasis, numBasis) = designMatrixRow;
            hGrad.row(2).tail(numBasis) = designMatrixRow;
            hGrads.push_back(hGrad);
        }

        GMMContainer<double> gmmContainer = GMMContainer<double>(keypoint, J, cameras, nu, designMatrix);
        gmmContainer.parameters.pi = pi;
        MC3DModel<double> mc3dModel = MC3DModel<double>(bspline, responsibilities, gmmContainer, hGrads, cameras);

        for (int i = 0; i < 1000; i++)
        {
            RowMatrix<double> theta = RowMatrix<double>::Random(numBasis, J * 3);

        std::function<double(RowMatrix<double> &)> fn([&](RowMatrix<double> &newTheta) -> double
        {
            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            return mc3dModel(newTheta.reshaped().transpose(), thetaGrad);
        });

        RowMatrix<double> finiteDiff = calcFiniteDiffMatrix(fn, theta, 1e-4);

            double fx = 0;
            Vector<double> thetaGrad = Vector<double>::Zero(numBasis * J * 3);
            RowMatrix<double> PWs = designMatrix * theta;
            mc3dModel.addSmoothnessPenalties(fx, thetaGrad, numBasis, PWs);
            RowMatrix<double> analyticDiff = thetaGrad.reshaped(numBasis, J * 3);

            double isApprox = analyticDiff.isApprox(finiteDiff, 1e-6);
            
            if (!isApprox)
            {
                std::cout << "Spline Smoothing Factor: " << gmmParam.splineSmoothingFactor << std::endl;
                std::cout << "Analytic: " << std::endl << analyticDiff.format(HeavyFormat) << std::endl;
                std::cout << "Finite: " << std::endl << finiteDiff.format(HeavyFormat) << std::endl;
            }

            ASSERT_TRUE(isApprox);
        }
    }
}

TEST(Derivatives, Projections)
{
    Camera<double> camera0 = initCamera0();
    WorldPoints<double> PWs(3, 1000);

    PWs.row(0) = (Vector<double>::Random(1000).array() + 2) * 1500;
    PWs.row(1) = (Vector<double>::Random(1000).array() + 2) * 500;
    PWs.row(2) = (Vector<double>::Random(1000).array() + 2) * 700;
    Vector<double> diffs(1000);

    for (int c = 0; c < PWs.cols(); c++)
    {
        WorldPoint<double> PW = PWs.col(c);

        double finiteDiff = calcFiniteDiff([&](double hdelta) {
            PW.array() += hdelta;
            double diff = camera0.projectSingle(PW).sum();
            PW.array() -= hdelta;
            return diff;
        });

        /*std::function<CameraPoints<double>(WorldPoint<double> &)> fn([&](WorldPoint<double> &newPW) -> CameraPoints<double>
        {
            return camera0.projectSingle(PW);
        });

        RowMatrix<double> finiteDiff = calcFiniteDiffMatrix(fn, PW, 1e-4);

        WorldPoint<double> finiteDiff = calcFiniteDiff([&](WorldPoint<double> &newPW) -> CameraPoints<double> {
            return camera0.projectSingle(PW).sum();
        }, PW, 1e-4);*/

        double analyticDiff = camera0.projectGrad(PW).sum();
        double diff = abs(analyticDiff - finiteDiff);
        diffs[c] = diff;

        if (diff >= 1e-3)
        {
            std::cout << "Diff: " << diff << std::endl;
            std::cout << "Analytic: " << analyticDiff << std::endl;
            std::cout << "Finite: " << finiteDiff << std::endl;
        }

        ASSERT_TRUE(diff <= 1e-3);
    }

    std::cout << "Max Diff: " << diffs.maxCoeff() << std::endl;
}