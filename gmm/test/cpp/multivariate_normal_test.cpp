#include <gtest/gtest.h>

#include "mc3d_common.h"
#include "multivariate_normal.h"

using namespace mc3d;

TEST(MultivariateNormal, Init)
{
    Tensor mean = torch::tensor({1.0, 0.5}, TensorRealTypeOption);
    Tensor covariance = torch::eye(2, TensorRealTypeOption);

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
    Tensor mean = torch::tensor({1.0, 0.5}, TensorRealTypeOption).reshape({2, 1});
    Tensor covariance = torch::eye(2, TensorRealTypeOption);

    MultivariateNormal<2> mvn(mean, covariance);

    Tensor x = torch::tensor({0.3, 0.7}, TensorRealTypeOption).reshape({2, 1});
    ASSERT_DOUBLE_EQ(mvn.pdf(x).item().toDouble(), 0.12210461930817082);

    x = torch::tensor({0.7, 0.3}, TensorRealTypeOption).reshape({2, 1});
    ASSERT_DOUBLE_EQ(mvn.pdf(x).item().toDouble(), 0.1491389188070974);

    x = torch::tensor({0.2, 0.5}, TensorRealTypeOption).reshape({2, 1});
    ASSERT_DOUBLE_EQ(mvn.pdf(x).item().toDouble(), 0.11557020867169786);
}

TEST(MultivariateNormal, PDFOfPoints)
{
    Tensor mean = torch::tensor({2982.75, -81.2951}, TensorRealTypeOption).reshape({2, 1});
    Tensor x = torch::tensor({1472.1, 311.79}, TensorRealTypeOption).reshape({2, 1});
    Tensor covariance = torch::eye(2, TensorRealTypeOption) * RealType(100000.0);

    MultivariateNormal<2> mvn(mean, covariance);

    double expected_P = 8.144330107692559e-12;

    ASSERT_TRUE(abs(mvn.pdf(x).item().toDouble() - expected_P) < 1e-10);
}