#include "mc3d_common.h"
#include "normal.h"
#include <gtest/gtest.h>

using namespace MC3D_TRECSIM;

TEST(Normal, Init)
{
    double mean{2};
    double variance{2.0};

    Normal<double> normal(mean, variance);

    ASSERT_DOUBLE_EQ(normal.getMean(), mean);
    ASSERT_DOUBLE_EQ(normal.getVariance(), variance);
}

TEST(Normal, DefaultInit)
{
    double mean{0};
    double variance{1.0};

    Normal<double> normal;

    ASSERT_DOUBLE_EQ(normal.getMean(), mean);
    ASSERT_DOUBLE_EQ(normal.getVariance(), variance);
}

TEST(Normal, PDFStandard)
{
    double mean{0.0};
    double variance{1.0};
    Normal<double> normal(mean, variance);

    Eigen::VectorX<double> x = Eigen::VectorX<double>::LinSpaced(101, -6.0, 6.0);

    for (size_t i = 0; i < x.size(); ++i)
    {
        const double pdf = normal.pdf(x[i]);
        const double logPdf = normal.logPdf(x[i]);

        std::cout << "x = " << x[i] << ", ";
        std::cout << "pdf = " << pdf << ", ";
        std::cout << "logPdf = " << logPdf << std::endl;

        ASSERT_DOUBLE_EQ(pdf, 1 / sqrt(2 * M_PI * variance) * exp(-pow((x[i] - mean), 2) / (2 * variance)));
        ASSERT_DOUBLE_EQ(logPdf, -std::log(sqrt(2 * M_PI * variance)) - pow((x[i] - mean), 2) / (2 * variance));
    }
}

TEST(Normal, PDFNonStandard)
{
    double mean{20.0};
    double variance{9.0};
    Normal<double> normal(mean, variance);

    Eigen::VectorX<double> x = Eigen::VectorX<double>::LinSpaced(101, 0, 40.0);

    for (size_t i = 0; i < x.size(); ++i)
    {
        const double pdf = normal.pdf(x[i]);
        const double logPdf = normal.logPdf(x[i]);

        std::cout << "x = " << x[i] << ", ";
        std::cout << "pdf = " << pdf << ", ";
        std::cout << "logPdf = " << logPdf << std::endl;

        ASSERT_DOUBLE_EQ(pdf, 1 / sqrt(2 * M_PI * variance) * exp(- pow((x[i] - mean), 2) / (2*variance)));
        ASSERT_DOUBLE_EQ(logPdf, -std::log(sqrt(2 * M_PI * variance)) - pow((x[i] - mean), 2) / (2*variance));
    }
}