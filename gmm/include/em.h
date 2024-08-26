#ifndef EM_H
#define EM_H

#include "config.h"
#include "mc3d_common.h"
#include "gmm_container.h"
#include "gmm_maximizer.h"

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    struct EMFitResult
    {
        GMMParameters<Scalar> parameters;
        RowMatrix<Scalar> designMatrix;
        RowMatrix<Scalar> responsibilities;
        Scalar diff;
        bool convergence;
        std::vector<int> niters;
        std::vector<bool> supports;

        EMFitResult() : parameters(),
                        designMatrix(RowMatrix<Scalar>::Zero(0, 0)),
                        responsibilities(Vector<Scalar>::Zero(0)),
                        diff(std::numeric_limits<Scalar>::infinity()),
                        convergence(false),
                        niters({}),
                        supports(0)
        { }
    };

    template <typename Scalar>
    class EM
    {
    public:
        EM(int maxIter, Scalar tol, GMMMaximizer<Scalar> &emMaximizer);

        inline RowMatrix<Scalar> expectation(GMMContainer<Scalar> &model);

        void expectation(GMMContainer<Scalar> &model, RowMatrix<Scalar> &responsibilites);

        inline void maximization(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilities, GMMMaximizationResult<Scalar> &gmmMaximizationResult);

        inline void operator()(GMMContainer<Scalar> &model, EMFitResult<Scalar> &fitResult);

        inline EMFitResult<Scalar> operator()(GMMContainer<Scalar> &model);

    private:
        int maxIter;
        Scalar tol;
        GMMMaximizer<Scalar> &emMaximizer;
    };
}
#include "em_impl.h"
#endif