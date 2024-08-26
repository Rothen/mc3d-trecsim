#ifndef GMM_MAXIMIZER_IMPL_H
#define GMM_MAXIMIZER_IMPL_H

#include "gmm_maximizer.h"

#include <set>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    GMMMaximizer<Scalar>::GMMMaximizer(const BSpline<Scalar> &spline, const RowMatrix<Scalar> &designMatrix, const std::vector<RowMatrix<Scalar>> &hGrads, std::vector<Camera<Scalar>> &cameras, const GMMParam<Scalar> &gmmParam, const LBFGSpp::LBFGSParam<Scalar> &lbfgsParam) :
        spline(spline),
        designMatrix(designMatrix),
        hGrads(hGrads),
        gmmParam(gmmParam),
        cameras(cameras),
        lbfgsSolver(LBFGSpp::LBFGSSolver<Scalar>(lbfgsParam))
    { }

    template <typename Scalar>
    void GMMMaximizer<Scalar>::updateSupport(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilites)
    {
        for (int c = 0; c < responsibilites.cols(); ++c)
        {
            std::set<size_t> supportedCameraIndizes;
            model.supports[c].supported = false;

            for (int r = responsibilites.rows() - 1; r >= std::max(0, int(responsibilites.rows() - gmmParam.responsibilityLookback)); --r)
            {
                if (responsibilites(r, c) >= gmmParam.responsibilitySupportThreshold)
                {
                    supportedCameraIndizes.insert(model.keyPoints[r].cameraIndex);
                }

                model.supports[c].supported = supportedCameraIndizes.size() >= gmmParam.numSupportCameras;

                if (model.supports[c].supported)
                {
                    break;
                }
            }

            model.supports[c].isNew = model.supports[c].isNew && !model.supports[c].supported;
        }
    }

    template <typename Scalar>
    inline void GMMMaximizer<Scalar>::operator()(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilities, GMMMaximizationResult<Scalar> &gmmMaximizationResult)
    {
        updateSupport(model, responsibilities);

        MC3DModel<Scalar> fun(spline, responsibilities, model, hGrads, cameras);

        Vector<Scalar> x = model.parameters.theta.reshaped().transpose();

        Scalar fx;

        gmmMaximizationResult.niter = lbfgsSolver.minimize(fun, x, fx);
        gmmMaximizationResult.parameters.theta = x.reshaped(spline.getNumBasis(), 3 * model.getNumHypothesis());
        gmmMaximizationResult.parameters.pi = responsibilities.colwise().sum() / responsibilities.sum();
    }

    template <typename Scalar>
    inline void GMMMaximizer<Scalar>::afterOptimization(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilities, const GMMMaximizationResult<Scalar> &gmmMaximizationResult)
    {
        for (Support &support : model.supports)
        {
            if (support.supported)
            {
                support.notSupportedSince = 0;
            }
            else
            {
                ++support.notSupportedSince;
            }
        }

#ifdef DEBUG_STATEMENTS
        std::cout << "Supports for keypoint " << model.KEYPOINT << ": ";
        for (const Support &support : model.supports)
        {
            std::cout << support.supported << " ";
        }
        std::cout << std::endl;
#endif
    }
}
#endif