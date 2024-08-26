#ifndef GMM_CONTAINER_IMPL_H
#define GMM_CONTAINER_IMPL_H

#include "gmm_container.h"
#include "frame.h"
#include "bspline.h"
#include "camera.h"
#include "multivariate_normal.h"

#include <LBFGS.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

namespace MC3D_TRECSIM
{
    Support::Support(int addedAt) :
        supported(false),
        notSupportedSince(0),
        isNew(true),
        addedAt(addedAt)
    { }

    template <typename Scalar>
    GMMContainer<Scalar>::GMMContainer(int KEYPOINT, const int &J, std::vector<Camera<Scalar>> cameras, Scalar nu, const RowMatrix<Scalar> &designMatrix) :
        KEYPOINT(KEYPOINT),
        keyPoints({}),
        meanPersonPerFrame(0),
        hypothesisChanged(false),
        enoughPoints(false),
        J(J),
        nu(nu),
        mvn(Eigen::Vector<double, 2>::Zero(), Eigen::Matrix<Scalar, 2, 2>::Identity(2, 2) * nu),
        designMatrix(designMatrix),
        cameras(cameras),
        parameters(ColMatrix<Scalar>::Zero(0, J*3), Vector<Scalar>::Zero(J)),
        supports(J, Support())
    { }

    template <typename Scalar>
    inline void GMMContainer<Scalar>::addKeypoint(Vector<Scalar> keypoint, Scalar time, size_t cameraIndex, unsigned int frameIndex)
    {
        keyPoints.push_back({keypoint, time, cameraIndex, frameIndex});
    }

    template <typename Scalar>
    void GMMContainer<Scalar>::dropFrame()
    {
        while (keyPoints.begin() != keyPoints.end() && keyPoints.begin()->frameIndex == 0)
        {
            keyPoints.erase(keyPoints.begin());
        }

        for (auto it = keyPoints.begin(); it != keyPoints.end(); ++it)
        {
            it->frameIndex -= 1;
        }

        for (auto &support : supports)
        {
            support.addedAt = std::max(0, support.addedAt - 1);
        }
    }

    template <typename Scalar>
    inline const int GMMContainer<Scalar>::getNumHypothesis() const
    {
        return J;
    }

    template <typename Scalar>
    inline const int GMMContainer<Scalar>::getNumValues() const
    {
        return keyPoints.size();
    }

    template <typename Scalar>
    inline Scalar GMMContainer<Scalar>::logProb(const int valueIndex, const int hypothesisIndex)
    {
        /*if (keyPoints[valueIndex].frameIndex < supports[hypothesisIndex].addedAt)
        {
            return -std::numeric_limits<Scalar>::max();
        }*/

        cameras[keyPoints[valueIndex].cameraIndex].projectSingle(
            (
                designMatrix.row(keyPoints[valueIndex].frameIndex) * parameters.theta.middleCols(hypothesisIndex * 3, 3)
            ).transpose(),
            tempCameraPoint
        );
        return mvn.logPdf(tempCameraPoint, keyPoints[valueIndex].keypoint) + std::log(
            std::max(parameters.pi(hypothesisIndex), std::numeric_limits<Scalar>::epsilon())
        );
    }
}
#endif