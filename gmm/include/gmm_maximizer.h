#ifndef GMM_MAXIMIZER_H
#define GMM_MAXIMIZER_H

#include "config.h"
#include "mc3d_common.h"
#include "frame.h"
#include "bspline.h"
#include "camera.h"
#include "gmm_container.h"
#include "multivariate_normal.h"
#include "mc3d_model.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "gmm_param.h"

#include <LBFGS.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>
#include <memory>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    class GMMMaximizationResult
    {
    public:
        GMMMaximizationResult() : niter(0), parameters() {}

        GMMParameters<Scalar> parameters;
        int niter;
    };

    template <typename Scalar>
    class GMMMaximizer
    {
    public:
        GMMMaximizer(const BSpline<Scalar> &spline, const RowMatrix<Scalar> &designMatrix, const std::vector<RowMatrix<Scalar>> &hGrads, std::vector<Camera<Scalar>> &cameras, const GMMParam<Scalar> &gmmParam, const LBFGSpp::LBFGSParam<Scalar> &lbfgsParam);

        void updateSupport(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilites);

        inline void operator()(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilities, GMMMaximizationResult<Scalar> &gmmMaximizationResult);

        inline void afterOptimization(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilities, const GMMMaximizationResult<Scalar> &gmmMaximizationResult);

    private:
        const BSpline<Scalar> &spline;
        const RowMatrix<Scalar> &designMatrix;
        const std::vector<RowMatrix<Scalar>> &hGrads;
        std::vector<Camera<Scalar>> &cameras;
        const GMMParam<Scalar> &gmmParam;
        LBFGSpp::LBFGSSolver<Scalar> lbfgsSolver;
    };
}
#include "gmm_maximizer_impl.h"
#endif