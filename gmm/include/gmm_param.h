#ifndef GMM_PARAM_H
#define GMM_PARAM_H

#include "config.h"
#include "mc3d_common.h"

#include <vector>
#include <tuple>
#include <iostream>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    class GMMParam
    {
    public:
        std::vector<int> KEYPOINTS;
        Scalar nu;
        unsigned int maxIter;
        Scalar keypointConfidenceThreshold;
        Scalar tol;
        Scalar splineDegree;
        Scalar splineKnotDelta;
        unsigned int maxFrameBuffer;
        bool autoManageTheta;
        bool autoManageHypothesis;
        bool copyLastThetas;
        Scalar splineSmoothingFactor;
        size_t numSupportCameras;
        size_t notSupportedSinceThreshold;
        size_t responsibilityLookback;
        Scalar responsibilitySupportThreshold;
        Scalar totalResponsibilitySupportThreshold;
        bool dragAlongUnsupportedKeyPoints;
        unsigned int seed;
        int minValidKeyPoints;

        GMMParam() : KEYPOINTS(std::vector<int>()),
                     nu(1.0),
                     maxIter(100),
                     keypointConfidenceThreshold(0.5),
                     tol(1.0),
                     splineDegree(3),
                     splineKnotDelta(500),
                     maxFrameBuffer(25),
                     autoManageTheta(false),
                     autoManageHypothesis(false),
                     copyLastThetas(false),
                     splineSmoothingFactor(0.0),
                     numSupportCameras(2),
                     notSupportedSinceThreshold(5),
                     responsibilityLookback(10),
                     responsibilitySupportThreshold(0.3),
                     totalResponsibilitySupportThreshold(0.3),
                     dragAlongUnsupportedKeyPoints(true),
                     seed(static_cast<unsigned>(time(0))),
                     minValidKeyPoints(0)
        {
            setSeed(seed);
        }

        unsigned int getSeed()
        {
            return seed;
        }

        void setSeed(unsigned int seed)
        {
            this->seed = seed;
            srand(seed);
        }

        friend std::ostream &operator<<(std::ostream &os, const GMMParam &obj)
        {
            std::cout << "GMMParams:" << std::endl;
            std::cout << "\tKEYPOINTS: ";
            for (int i = 0; i < obj.KEYPOINTS.size()-1; i++)
            {
                std::cout << obj.KEYPOINTS[i] << ", ";
            }
            if (obj.KEYPOINTS.size() > 0)
            {
                std::cout << obj.KEYPOINTS[obj.KEYPOINTS.size()-1];
            }
            std::cout << std::endl;
            std::cout << "\t" << "nu: " << obj.nu << std::endl;
            std::cout << "\t" << "maxIter: " << obj.maxIter << std::endl;
            std::cout << "\t" << "keypointConfidenceThreshold: " << obj.keypointConfidenceThreshold << std::endl;
            std::cout << "\t" << "tol: " << obj.tol << std::endl;
            std::cout << "\t" << "splineDegree: " << obj.splineDegree << std::endl;
            std::cout << "\t" << "splineKnotDelta: " << obj.splineKnotDelta << std::endl;
            std::cout << "\t" << "maxFrameBuffer: " << obj.maxFrameBuffer << std::endl;
            std::cout << "\t" << "autoManageTheta: " << obj.autoManageTheta << std::endl;
            std::cout << "\t" << "autoManageHypothesis: " << obj.autoManageHypothesis << std::endl;
            std::cout << "\t" << "copyLastThetas: " << obj.copyLastThetas << std::endl;
            std::cout << "\t" << "splineSmoothingFactor: " << obj.splineSmoothingFactor << std::endl;
            std::cout << "\t" << "numSupportCameras: " << obj.numSupportCameras << std::endl;
            std::cout << "\t" << "notSupportedSinceThreshold: " << obj.notSupportedSinceThreshold << std::endl;
            std::cout << "\t" << "responsibilityLookback: " << obj.responsibilityLookback << std::endl;
            std::cout << "\t" << "responsibilitySupportThreshold: " << obj.responsibilitySupportThreshold << std::endl;
            std::cout << "\t" << "totalResponsibilitySupportThreshold: " << obj.totalResponsibilitySupportThreshold << std::endl;
            std::cout << "\t" << "dragAlongUnsupportedKeyPoints: " << obj.dragAlongUnsupportedKeyPoints << std::endl;
            std::cout << "\t" << "minValidKeyPoints: " << obj.minValidKeyPoints << std::endl;
            return os;
        }
    };
}
#endif