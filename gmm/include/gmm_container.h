#ifndef GMM_CONTAINER_H
#define GMM_CONTAINER_H

#include "config.h"
#include "mc3d_common.h"
#include "frame.h"
#include "bspline.h"
#include "camera.h"
#include "multivariate_normal.h"
#include "normal.h"

#include <LBFGS.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <functional>
#include <math.h>

namespace MC3D_TRECSIM
{
    struct Support
    {
        Support(int addedAt = 0);

        bool supported;
        size_t notSupportedSince;
        bool isNew;
        int addedAt;
    };

    template <typename Scalar>
    struct KeyPoint
    {
        Vector<Scalar> keypoint;
        Scalar time;
        size_t cameraIndex;
        unsigned int frameIndex;
    };

    template <typename Scalar>
    class GMMParameters
    {
    public:
        ColMatrix<Scalar> theta;
        Vector<Scalar> pi;

        GMMParameters(ColMatrix<Scalar> theta = ColMatrix<Scalar>::Zero(0, 0), Vector<Scalar> pi = Vector<Scalar>::Zero(0)) : theta(theta), pi(pi) {}

        inline Scalar operator-(const GMMParameters<Scalar> &other) const
        {
            return (theta - other.theta).norm();
        }
    };

    template <typename Scalar>
    class GMMContainer
    {
    public:
        std::vector<KeyPoint<Scalar>> keyPoints;
        int KEYPOINT;
        Scalar meanPersonPerFrame;
        bool hypothesisChanged;
        MultivariateNormal<Scalar, 2> mvn;
        const RowMatrix<Scalar> &designMatrix;
        const int &J;
        std::vector<Camera<Scalar>> cameras;
        GMMParameters<Scalar> parameters;
        Scalar nu;
        std::vector<Support> supports;

        GMMContainer(int KEYPOINT = 0, const int &J = 0, std::vector<Camera<Scalar>> cameras = std::vector<Camera<Scalar>>(), Scalar nu = 1.0, const RowMatrix<Scalar> &designMatrix = RowMatrix<Scalar>::Zero(0, 0));

        inline void addKeypoint(Vector<Scalar> keypoint, Scalar time, size_t cameraIndex, unsigned int frameIndex);

        void dropFrame();

        inline const int getNumHypothesis() const;
        
        inline const int getNumValues() const;

        inline Scalar logProb(const int valueIndex, const int hypothesisIndex);

    private:
        bool enoughPoints;
        CameraPoint<Scalar> tempCameraPoint;
    };
}
#include "gmm_container_impl.h"
#endif