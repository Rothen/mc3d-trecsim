#ifndef HYPOTHESIS_MANAGER_H
#define HYPOTHESIS_MANAGER_H

#include "config.h"
#include "mc3d_common.h"
#include "em.h"
#include "frame.h"
#include "camera.h"
#include "gmm_param.h"

#include <Eigen/Core>
#include <vector>
#include <map>
#include <tuple>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    struct Hypothesis
    {
        Hypothesis();

        CameraPoint<Scalar> lastMeanPoint;
        size_t notSeenSince;
        size_t inViewSince;
    };

    template <typename Scalar>
    struct CameraHypotheses
    {
        CameraHypotheses();

        std::vector<std::tuple<CameraPoint<Scalar>, Scalar>> hypothesisPoints;
        size_t nPeopleInFrame;
    };

    template <typename Scalar>
    class HypothesisManager
    {
    public:
        HypothesisManager(const std::vector<Camera<Scalar>> &cameras, const GMMParam<Scalar> &gmmParam);

        void update(const Frame<Scalar> &frame, std::vector<CameraPoint<Scalar>> &newHypothesisPoints, std::vector<size_t> &existingHypothesisIndizes);

        void calculateMeanPoints(std::vector<CameraPoint<Scalar>> &currentMeanPoints, const std::vector<RowMatrix<Scalar>> &people);

        void calculateMeanPointsWeighted(std::vector<CameraPoint<Scalar>> &currentMeanPoints, const std::vector<RowMatrix<Scalar>> &people);

        inline void matchupHypotheses(std::vector<std::tuple<CameraPoint<Scalar>, Scalar>> &newHypothesisPoints, const std::vector<CameraPoint<Scalar>> &currentMeanPoints, const std::vector<std::tuple<CameraPoint<Scalar>, Scalar>> &hypothesisPoints);

        inline void removeHypothesis(int index);

    private:
        const std::vector<Camera<Scalar>> &cameras;
        const GMMParam<Scalar> &gmmParam;
        std::vector<CameraHypotheses<Scalar>> cameraHypotheses;
        size_t currentNumberOfHypothesis;
        size_t maxNPeopleInFrameCameraIndex = 0;
    };
}
#include "hypothesis_manager_impl.h"
#endif