#ifndef MC3D_MODEL_IMPL_H
#define MC3D_MODEL_IMPL_H

#include "mc3d_model.h"
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
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    MC3DModel<Scalar>::MC3DModel(const BSpline<Scalar>& spline, const RowMatrix<Scalar>& responsibilities, const GMMContainer<Scalar>& gmmContainer, const std::vector<RowMatrix<Scalar>>& hGrads, std::vector<Camera<Scalar>>& cameras) :
        spline(spline),
        responsibilities(responsibilities),
        gmmContainer(gmmContainer),
        hGrads(hGrads),
        cameras(cameras)
    {

        nu = gmmContainer.nu;
        minusLog2PiMinLogNu = -log(2 * M_PI) - log(nu);
        factor = 1.0 / (2.0 * nu);
    }

    template <typename Scalar>
    Scalar MC3DModel<Scalar>::operator()(const Vector<Scalar> &theta, Vector<Scalar> &thetaGrad)
    {
        int numBasis{spline.getNumBasis()};
        int numHypothesis{gmmContainer.getNumHypothesis()};
        Scalar fx{0.0};
        RowMatrix<Scalar> PWs = gmmContainer.designMatrix * theta.reshaped(numBasis, numHypothesis * 3);

    #if defined(DEBUG_STATEMENTS) && defined(INTENSE_DEBUG_STATEMENTS)
        std::cout << "Retransformed Theta: " << theta.reshaped(numBasis, numHypothesis * 3) << std::endl;
    #endif

        thetaGrad = Vector<Scalar>::Zero(thetaGrad.size());

        for (int j = 0; j < numHypothesis; ++j)
        {
            if (!gmmContainer.supports[j].supported)
            {
                continue;
            }
            for (int n = 0; n < gmmContainer.keyPoints.size(); ++n)
            {
                tempPW << PWs.row(gmmContainer.keyPoints[n].frameIndex).segment(j * 3, 3).transpose();
                cameras[gmmContainer.keyPoints[n].cameraIndex].projectSingle(tempPW, pIDest);
                cameras[gmmContainer.keyPoints[n].cameraIndex].projectGrad(tempPW, PWGradDest);
                tempDist << (gmmContainer.keyPoints[n].keypoint - pIDest);

                fx -= responsibilities(n, j) * (minusLog2PiMinLogNu - factor * tempDist.dot(tempDist));

                thetaGrad.segment(j * numBasis * 3, numBasis * 3) -= responsibilities(n, j) / nu * tempDist.transpose() * PWGradDest * hGrads.at(gmmContainer.keyPoints[n].frameIndex);
            }
        }

        addSmoothnessPenalties(fx, thetaGrad, numBasis, PWs);

        return fx;
    }

    template <typename Scalar>
    inline void MC3DModel<Scalar>::addSmoothnessPenalties(Scalar &fx, Vector<Scalar> &thetaGrad, const int numBasis, const RowMatrix<Scalar> &PWs)
    {
        const int offset{int(gmmContainer.designMatrix.rows()) - numBasis + 2};
        const int numBasisTimes3 = numBasis * 3;

        for (int j = 0; j < gmmContainer.getNumHypothesis(); ++j)
        {
            if (!gmmContainer.supports[j].supported)
            {
                continue;
            }
            for (int n = 0; n < numBasis - 2; ++n)
            {
                tempPW << PWs.row(offset + n).segment(j * 3, 3).transpose();
                fx += tempPW.dot(tempPW);
                thetaGrad.segment(j * numBasisTimes3, numBasisTimes3) += 2 * tempPW.transpose() * hGrads.at(offset + n);
            }
        }
    }
}
#endif