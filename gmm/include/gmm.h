#ifndef GMM_H
#define GMM_H

#include "config.h"
#include "mc3d_common.h"
#include "em.h"
#include "frame.h"
#include "bspline.h"
#include "camera.h"
#include "gmm_container.h"
#include "multivariate_normal.h"
#include "mc3d_model.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "hypothesis_manager.h"
#include "gmm_param.h"
#include "gmm_maximizer.h"

#include <LBFGS.h>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>
#include <memory>
#include <functional>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    class GMM
    {
    public:
        int J;
        std::vector<Camera<Scalar>> cameras;
        std::vector<Frame<Scalar>> frames;
        Vector<Scalar> times;
        BSpline<Scalar> spline;
        std::map<int, GMMContainer<Scalar>> gmmContainers;
        GMMParam<Scalar> gmmParam;
        LBFGSpp::LBFGSParam<Scalar> lbfgsParam;
        RowMatrix<Scalar> designMatrix;
        std::vector<RowMatrix<Scalar>> hGrads;
        HypothesisManager<Scalar> hypothesisManager;
        GMMMaximizer<Scalar> gmmMaximizer;
        EM<Scalar> em;
        std::vector<unsigned long> hypothesisIds;

        GMM(int J, std::vector<Camera<Scalar>> camerasE, GMMParam<Scalar> gmmParamE, LBFGSpp::LBFGSParam<Scalar> lbfgsParamE);

        inline WorldPoints<Scalar> h(int i, const ColMatrix<Scalar>& theta);

        void prepareGMMContainers(const std::map<int, ColMatrix<Scalar>>& initialThetas = std::map<int, ColMatrix<Scalar>>(), const std::map<int, Vector<Scalar>>& initialPis = std::map<int, Vector<Scalar>>());

        void prepareDesignMatrix();

        inline void prepareCalculations(const std::map<int, ColMatrix<Scalar>>& initialThetas = std::map<int, ColMatrix<Scalar>>(), const std::map<int, Vector<Scalar>>& initialPis = std::map<int, Vector<Scalar>>());

        std::map<int, EMFitResult<Scalar>> fit(const std::map<int, ColMatrix<Scalar>> &initialThetas = std::map<int, ColMatrix<Scalar>>(), const std::map<int, Vector<Scalar>> &initialPis = std::map<int, Vector<Scalar>>());

        void removeHypothesis(int index);

        void removeHypothesisKeyPoints(int index, std::map<int, EMFitResult<Scalar>> &fitResults);

        void addHypothesis(const WorldPoint<Scalar> &worldPoint);

        inline void addHypothesis();

        void addFrame(Frame<Scalar> &frame);

        void filterPeople(Frame<Scalar> &frame);

        void addFrameToGMMContainers(const Frame<Scalar> &frame);

        void addTimeToBSpline(Scalar time);

        void fillupThetas();

        template <typename FriendScalar>
        friend GMM<FriendScalar> &operator<<(GMM<FriendScalar> &self, Frame<FriendScalar> &frame);

    private:
        WorldPoint<Scalar> tempSupportedMeanPoint = WorldPoint<Scalar>::Zero(3);
        unsigned long currentHypothesisId = 0;

        void initGMMContainers();
    };
}
#include "gmm_impl.h"
#endif