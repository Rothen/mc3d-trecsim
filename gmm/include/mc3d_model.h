#ifndef MC3D_MODEL_H
#define MC3D_MODEL_H

#include "config.h"
#include "mc3d_common.h"
#include "frame.h"
#include "bspline.h"
#include "camera.h"
#include "gmm_container.h"
#include "multivariate_normal.h"
#include "gmm_param.h"

#include <LBFGS.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <memory>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    class MC3DModel
    {
    public:
        MC3DModel(const BSpline<Scalar> &spline, const RowMatrix<Scalar> &responsibilities, const GMMContainer<Scalar> &gmmContainer, const std::vector<RowMatrix<Scalar>> &hGrads, std::vector<Camera<Scalar>> &cameras);

        Scalar operator()(const Vector<Scalar> &theta, Vector<Scalar> &thetaGrad);

        inline void addLengthRestrictions(Scalar &fx, Vector<Scalar> &thetaGrad, const int numBasis, const int numHypothesis, const RowMatrix<Scalar> &PWs);

        inline void addSmoothnessPenalties(Scalar &fx, Vector<Scalar> &thetaGrad, const int numBasis, const RowMatrix<Scalar> &PWs);

    private:
        const BSpline<Scalar> &spline;
        RowMatrix<Scalar> responsibilities;
        const GMMContainer<Scalar> &gmmContainer;
        const std::vector<RowMatrix<Scalar>> &hGrads;
        std::vector<Camera<Scalar>> cameras;

        // For calculations
        Scalar nu;
        Scalar minusLog2PiMinLogNu;
        Scalar factor;
        CameraPoint<Scalar> pIDest;
        CameraPointGrad<Scalar> PWGradDest;
        WorldPoint<Scalar> tempPW;
        WorldPoint<Scalar> tempPW2;
        CameraPoint<Scalar> tempDist;
    };
}
#include "mc3d_model_impl.h"
#endif