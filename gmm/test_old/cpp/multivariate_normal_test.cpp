#include "mc3d_common.h"
#include "multivariate_normal.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using namespace MC3D_TRECSIM;

TEST(MultivariateNormal, Init)
{
    Eigen::Vector<double, 2> mean(2);
    mean << 1.0, 0.5;
    Eigen::Matrix<double, 2, 2> covariance(2, 2);
    covariance << 1.0, 0.0, 0.0, 1.0;

    MultivariateNormal<double, 2> mvn(mean, covariance);

    EXPECT_TRUE(mvn.getMean().isApprox(mean));
    EXPECT_TRUE(mvn.getCovariance().isApprox(covariance));
}

TEST(MultivariateNormal, DefaultInit)
{
    Eigen::Vector<double, 2> mean(2);
    mean << 0.0, 0.0;
    Eigen::Matrix<double, 2, 2> covariance(2, 2);
    covariance << 1.0, 0.0, 0.0, 1.0;

    MultivariateNormal<double, 2> mvn;

    EXPECT_TRUE(mvn.getMean().isApprox(mean));
    EXPECT_TRUE(mvn.getCovariance().isApprox(covariance));
}

TEST(MultivariateNormal, PDF)
{
    Eigen::Vector<double, 2> mean(2);
    mean << 1.0, 0.5;
    Eigen::Matrix<double, 2, 2> covariance(2, 2);
    covariance << 1.0, 0.0, 0.0, 1.0;

    MultivariateNormal<double, 2> mvn(mean, covariance);

    Eigen::Vector<double, 2> x(2);
    x << 0.3, 0.7;
    double P = mvn.pdf(x);
    double expectedP = 0.12210461930817082;
    ASSERT_DOUBLE_EQ(P, expectedP);

    x << 0.7, 0.3;
    P = mvn.pdf(x);
    expectedP = 0.1491389188070974;
    ASSERT_DOUBLE_EQ(P, expectedP);

    x << 0.2, 0.5;
    P = mvn.pdf(x);
    expectedP = 0.11557020867169786;
    ASSERT_DOUBLE_EQ(P, expectedP);
}

TEST(MultivariateNormal, PDFOfPoints)
{
    Eigen::Vector<double, 2> mean(2);
    mean << 2982.75, -81.2951;
    Eigen::Vector<double, 2> x(2);
    x    << 1472.1,   311.79;
    Eigen::Matrix<double, 2, 2> covariance(2, 2);
    covariance << 1.0, 0.0, 0.0, 1.0;
    covariance *= 100000;

    MultivariateNormal<double, 2> mvn(mean, covariance);

    double P = mvn.pdf(x);
    double expectedP = 8.144330107692559e-12;
    ASSERT_TRUE(abs(P - expectedP) < 1e-10);
}