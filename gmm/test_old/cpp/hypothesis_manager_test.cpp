#include "mc3d_common.h"
#include "hypothesis_manager.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using namespace MC3D_TRECSIM;

TEST(HypothesisManager, CalculateMeanPoints)
{
    std::vector<CameraPoint<double>> currentMeanPoints;
    std::vector<RowMatrix<double>> kpts{
        (RowMatrix<double>(5, 3) << 1.0, 10.0, 0.5, 2.0, 20.0, 0.7, 3.0, 30.0, 0.4, 4.0, 40.0, 0.8, 5.0, 50.0, 0.9).finished(),
        (RowMatrix<double>(6, 3) << 1.0, 10.0, 0.5,   2.0, 20.0, 0.4,   3.0, 30.0, 0.6,   4.0, 40.0, 0.8,   5.0, 50.0, 0.1, 6.0, 60.0, 1.0).finished()
    };

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = std::vector<int>{0, 1, 2, 3, 4};
    HypothesisManager<double> hypothesisManager{std::vector<Camera<double>>{}, gmmParam};
    hypothesisManager.calculateMeanPoints(currentMeanPoints, kpts);

    std::cout << "expected: [3.0, 30.0], [8/3, 80/3]" << std::endl;
    for (auto &point : currentMeanPoints)
    {
        std::cout << point.transpose() << std::endl;
    }

    ASSERT_TRUE(currentMeanPoints[0].isApprox((CameraPoint<double>(2) << 3.0, 30.0).finished()));
    ASSERT_TRUE(currentMeanPoints[1].isApprox((CameraPoint<double>(2) << double(8.0/3.0), double(80.0/3.0)).finished()));
}

TEST(HypothesisManager, CalculateMeanPointsWeighted)
{
    std::vector<CameraPoint<double>> currentMeanPoints;
    std::vector<RowMatrix<double>> kpts{
        (RowMatrix<double>(5, 3) << 1.0, 10.0, 0.5, 2.0, 20.0, 0.7, 3.0, 30.0, 0.4, 4.0, 40.0, 0.8, 5.0, 50.0, 0.9).finished(),
        (RowMatrix<double>(6, 3) << 1.0, 10.0, 0.3, 2.0, 20.0, 0.7, 3.0, 30.0, 0.6, 4.0, 40.0, 0.8, 5.0, 50.0, 0.1, 6.0, 60.0, 1.0).finished()};

    GMMParam<double> gmmParam;
    gmmParam.KEYPOINTS = std::vector<int>{0, 1, 2, 3, 4};
    HypothesisManager<double> hypothesisManager{std::vector<Camera<double>>{}, gmmParam};
    hypothesisManager.calculateMeanPointsWeighted(currentMeanPoints, kpts);

    std::cout << "expected: [2.16, 21.6], [1.44, 14.4]" << std::endl;
    for (auto &point : currentMeanPoints)
    {
        std::cout << point.transpose() << std::endl;
    }

    ASSERT_TRUE(currentMeanPoints[0].isApprox((CameraPoint<double>(2) << 2.16, 21.6).finished()));
    ASSERT_TRUE(currentMeanPoints[1].isApprox((CameraPoint<double>(2) << 1.44, 14.4).finished()));
}

/*TEST(HypothesisManager, MatchupHypotheses)
{
    HypothesisManager<double> hypothesisManager{std::vector<Camera<double>>{}, GMMParam<double>{}};

    std::vector<CameraPoint<double>> lastMeanPoints{};
    lastMeanPoints.push_back((CameraPoint<double>(2) << 0.0, 0.0).finished());
    lastMeanPoints.push_back((CameraPoint<double>(2) << 1000.0, 1000.0).finished());

    std::vector<CameraPoint<double>> currentMeanPoints{};
    currentMeanPoints.push_back((CameraPoint<double>(2) << 500.0, 500.0).finished());
    currentMeanPoints.push_back((CameraPoint<double>(2) << 1001.0, 1001.0).finished());

    std::vector<CameraPoint<double>> newHypothesisPoints{};
    std::vector<size_t> existingHypothesisIndizes{};
    std::vector<size_t> oldHypothesisIndizes{};

    hypothesisManager.matchupHypotheses(newHypothesisPoints, existingHypothesisIndizes, oldHypothesisIndizes, currentMeanPoints, lastMeanPoints);

    ASSERT_EQ(newHypothesisPoints.size(), 1);
    ASSERT_EQ(existingHypothesisIndizes.size(), 1);
    ASSERT_EQ(oldHypothesisIndizes.size(), 1);

    ASSERT_TRUE(newHypothesisPoints[0].isApprox((CameraPoint<double>(2) << 500.0, 500.0).finished()));
    ASSERT_EQ(existingHypothesisIndizes[0], 1);
    ASSERT_EQ(oldHypothesisIndizes[0], 0);
}*/