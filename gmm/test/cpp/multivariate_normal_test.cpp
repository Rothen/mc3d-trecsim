#include <gtest/gtest.h>

#include "mc3d_common.h"
#include "multivariate_normal.h"

using namespace mc3d;

TEST(MultivariateNormal, Init)
{
    Tensor mean = torch::tensor({1.0, 0.5}, torch::dtype(torch::kDouble));
    Tensor covariance = torch::eye(2, torch::dtype(torch::kDouble));

    MultivariateNormal<2> mvn(mean, covariance);

    EXPECT_TRUE(mvn.get_mean().allclose(mean));
    EXPECT_TRUE(mvn.get_covariance().allclose(covariance));
}

TEST(MultivariateNormal, DefaultInit)
{
    Tensor mean = torch::tensor({0.0, 0.0});
    Tensor covariance = torch::eye(2);

    MultivariateNormal<2> mvn;

    EXPECT_TRUE(mvn.get_mean().allclose(mean));
    EXPECT_TRUE(mvn.get_covariance().allclose(covariance));
}

TEST(MultivariateNormal, PDF)
{
    Tensor mean = torch::tensor({1.0, 0.5});
    Tensor covariance = torch::eye(2);

    MultivariateNormal<2> mvn(mean, covariance);

    Tensor x = torch::tensor({0.3, 0.7});
    Tensor P = mvn.pdf(x);
    ASSERT_DOUBLE_EQ(P.item().toDouble(), 0.12210461930817082);

    x = torch::tensor({0.7, 0.3});
    P = mvn.pdf(x);
    ASSERT_DOUBLE_EQ(P.item().toDouble(), 0.1491389188070974);

    x = torch::tensor({0.2, 0.5});
    P = mvn.pdf(x);
    ASSERT_DOUBLE_EQ(P.item().toDouble(), 0.11557020867169786);
}

/*TEST(MultivariateNormal, PDFOfPoints)
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
}*/