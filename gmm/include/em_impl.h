#ifndef EM_IMPL_H
#define EM_IMPL_H

#include "em.h"

#include <set>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    EM<Scalar>::EM(int maxIter, Scalar tol, GMMMaximizer<Scalar> &emMaximizer) : 
        maxIter(maxIter),
        tol(tol),
        emMaximizer(emMaximizer) {}

    template <typename Scalar>
    inline RowMatrix<Scalar> EM<Scalar>::expectation(GMMContainer<Scalar> &model)
    {
        RowMatrix<Scalar> responsibilites(model.getNumValues(), model.getNumHypothesis());
        expectation(model, responsibilites);
        return responsibilites;
    }

    template <typename Scalar>
    void EM<Scalar>::expectation(GMMContainer<Scalar> &model, RowMatrix<Scalar> &responsibilites)
    {
        RowMatrix<Scalar> logProbs(model.getNumValues(), model.getNumHypothesis());

        for (int n = 0; n < model.getNumValues(); n++)
        {
            for (int j = 0; j < model.getNumHypothesis(); j++)
            {
                logProbs(n, j) = model.logProb(n, j);
            }
        }

        Vector<Scalar> sumExps(model.getNumValues());
        Vector<Scalar> max = logProbs.rowwise().maxCoeff();

        for (int n = 0; n < model.getNumValues(); ++n)
        {
            sumExps(n) = Scalar((logProbs.row(n).array() - max(n)).exp().sum());
        }

        for (int n = 0; n < model.getNumValues(); ++n)
        {
            for (int j = 0; j < model.getNumHypothesis(); ++j)
            {
                responsibilites(n, j) = std::exp(logProbs(n, j) - max(n)) / sumExps(n);
            }
        }
    }

    template <typename Scalar>
    inline void EM<Scalar>::maximization(GMMContainer<Scalar> &model, const RowMatrix<Scalar> &responsibilities, GMMMaximizationResult<Scalar> &gmmMaximizationResult)
    {
        emMaximizer(model, responsibilities, gmmMaximizationResult);
    }

    template <typename Scalar>
    inline EMFitResult<Scalar> EM<Scalar>::operator()(GMMContainer<Scalar> &model)
    {
        EMFitResult<Scalar> fitResult;
        (*this)(model, fitResult);
        return fitResult;
    }

    template <typename Scalar>
    inline void EM<Scalar>::operator()(GMMContainer<Scalar> &model, EMFitResult<Scalar> &fitResult)
    {
        GMMMaximizationResult<Scalar> gmmMaximizationResult;

        for (int iter = 0; iter < maxIter; ++iter)
        {
            GMMParameters<Scalar> oldParameters = model.parameters;
            expectation(model, fitResult.responsibilities);

#ifdef DEBUG_STATEMENTS
            std::cout << "Responsibilities: " << fitResult.responsibilities << std::endl;
#endif

            try
            {
                maximization(model, fitResult.responsibilities, gmmMaximizationResult);

                model.parameters = gmmMaximizationResult.parameters;
                fitResult.diff = oldParameters - gmmMaximizationResult.parameters;
                fitResult.parameters = gmmMaximizationResult.parameters;
                fitResult.niters.push_back(gmmMaximizationResult.niter);

                if (fitResult.diff <= tol)
                {
                    fitResult.convergence = true;
                    break;
                }
            }
            catch (const std::runtime_error& e)
            {
                fitResult.convergence = false;
            #ifdef DEBUG_STATEMENTS
                std::cout << e.what() << std::endl;
            #endif
                break;
            }
        }

        emMaximizer.afterOptimization(model, fitResult.responsibilities, gmmMaximizationResult);
    }
};
#endif