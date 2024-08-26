#ifndef HYPOTHESIS_MANAGER_IMPL_H
#define HYPOTHESIS_MANAGER_IMPL_H

#include "hypothesis_manager.h"

#include <limits>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    Hypothesis<Scalar>::Hypothesis() :
        lastMeanPoint(CameraPoint<Scalar>::Zero(2)),
        notSeenSince(0),
        inViewSince(0)
    { }

    template <typename Scalar>
    CameraHypotheses<Scalar>::CameraHypotheses() :
        hypothesisPoints(std::vector<std::tuple<CameraPoint<Scalar>, Scalar>>()),
        nPeopleInFrame(0)
    { }

    template <typename Scalar>
    HypothesisManager<Scalar>::HypothesisManager(const std::vector<Camera<Scalar>> &cameras, const GMMParam<Scalar> &gmmParam) :
        cameras(cameras),
        gmmParam(gmmParam),
        cameraHypotheses(std::vector<CameraHypotheses<Scalar>>(cameras.size())),
        currentNumberOfHypothesis(0)
    {
    }

    template <typename Scalar>
    struct greater_than_key
    {
        inline bool operator()(const std::tuple<CameraPoint<Scalar>, Scalar> &tuple1, const std::tuple<CameraPoint<Scalar>, Scalar> &tuple2)
        {
            return (std::get<Scalar>(tuple1) > std::get<Scalar>(tuple2));
        }
    };

    template <typename Scalar>
    void HypothesisManager<Scalar>::update(const Frame<Scalar> &frame, std::vector<CameraPoint<Scalar>> &newHypothesisPoints, std::vector<size_t> &existingHypothesisIndizes)
    {
        std::vector<std::tuple<CameraPoint<Scalar>, Scalar>> tempNewHypothesisPoints;
        std::vector<CameraPoint<Scalar>> meanPoints;

        calculateMeanPoints(meanPoints, frame.kpts);
        if (meanPoints.size() == 0)
        {
            return;
        }

        matchupHypotheses(tempNewHypothesisPoints, meanPoints, cameraHypotheses[frame.cameraIndex].hypothesisPoints);
        cameraHypotheses[frame.cameraIndex].hypothesisPoints = tempNewHypothesisPoints;

        cameraHypotheses[frame.cameraIndex].nPeopleInFrame = frame.kpts.size();

        size_t maxNPeopleInFrameCameraIndex = 0;
        CameraHypotheses<Scalar> &maxNPeopleInFrameCamera = cameraHypotheses[maxNPeopleInFrameCameraIndex];

        for (size_t i = 1; i < cameraHypotheses.size(); ++i)
        {
            if (cameraHypotheses[i].nPeopleInFrame > maxNPeopleInFrameCamera.nPeopleInFrame)
            {
                maxNPeopleInFrameCamera = cameraHypotheses[i];
                maxNPeopleInFrameCameraIndex = i;
            }
        }

        // std::cout << "maxNPeopleInFrameCameraIndex: " << maxNPeopleInFrameCameraIndex << std::endl;

        if (maxNPeopleInFrameCamera.nPeopleInFrame > currentNumberOfHypothesis)
        {
            for (size_t i = 0; i < maxNPeopleInFrameCamera.nPeopleInFrame - currentNumberOfHypothesis; ++i)
            {
                newHypothesisPoints.push_back(std::get<CameraPoint<Scalar>>(cameraHypotheses[frame.cameraIndex].hypothesisPoints[i]));
            }

            currentNumberOfHypothesis = maxNPeopleInFrameCamera.nPeopleInFrame;
        }
    }

    template <typename Scalar>
    inline void HypothesisManager<Scalar>::removeHypothesis(int index)
    {
        currentNumberOfHypothesis--;
    }

    template <typename Scalar>
    void HypothesisManager<Scalar>::calculateMeanPoints(std::vector<CameraPoint<Scalar>> &target, const std::vector<RowMatrix<Scalar>> &people)
    {
        for (const RowMatrix<Scalar> &person : people)
        {
            CameraPoint<Scalar> meanPoint = CameraPoint<Scalar>::Zero(2);
            int nKeypoints = 0;

            for (const int &KEYPOINT : gmmParam.KEYPOINTS)
            {
                if (person(KEYPOINT, 2) >= gmmParam.keypointConfidenceThreshold)
                {
                    meanPoint += person.row(KEYPOINT).head(2);
                    ++nKeypoints;
                }
            }

            if (nKeypoints > 0)
            {
                meanPoint /= nKeypoints;
                target.push_back(meanPoint);
            }
        }
    }

    template <typename Scalar>
    void HypothesisManager<Scalar>::calculateMeanPointsWeighted(std::vector<CameraPoint<Scalar>> &target, const std::vector<RowMatrix<Scalar>> &people)
    {
        for (const RowMatrix<Scalar> &person : people)
        {
            CameraPoint<Scalar> meanPoint = CameraPoint<Scalar>::Zero(2);
            int nKeypoints = 0;

            for (const int &KEYPOINT : gmmParam.KEYPOINTS)
            {
                meanPoint += person.row(KEYPOINT).head(2) * person(KEYPOINT, 2);
                ++nKeypoints;
            }

            if (nKeypoints > 0)
            {
                meanPoint /= nKeypoints;
                target.push_back(meanPoint);
            }
        }
    }

    template <typename Scalar>
    inline void HypothesisManager<Scalar>::matchupHypotheses(std::vector<std::tuple<CameraPoint<Scalar>, Scalar>> &newHypothesisPoints, const std::vector<CameraPoint<Scalar>> &meanPoints, const std::vector<std::tuple<CameraPoint<Scalar>, Scalar>> &hypothesisPoints)
    {
        for (const CameraPoint<Scalar> &currentMeanPoint : meanPoints)
        {
            Scalar minDist{std::numeric_limits<Scalar>::infinity()};

            for (const std::tuple<CameraPoint<Scalar>, Scalar> &hypothesisPoint : hypothesisPoints)
            {
                Scalar dist{(currentMeanPoint - std::get<CameraPoint<Scalar>>(hypothesisPoint)).norm()};

                if (dist < minDist)
                {
                    minDist = dist;
                }
            }

            newHypothesisPoints.push_back({currentMeanPoint, minDist});
        }

        std::sort(newHypothesisPoints.begin(), newHypothesisPoints.end(), greater_than_key<Scalar>());
    }
};
#endif