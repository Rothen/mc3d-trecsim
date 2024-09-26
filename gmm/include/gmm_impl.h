#ifndef GMM_IMPL_H
#define GMM_IMPL_H

#include "config.h"
#include "mc3d_common.h"
#include "gmm.h"
#include "frame.h"
#include "bspline.h"
#include "camera.h"
#include "gmm_container.h"
#include "multivariate_normal.h"
#include "mc3d_model.h"
#include "spdlog/sinks/basic_file_sink.h"
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
#include <cstdlib>
#include <iostream>
#include <ctime>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    Scalar LO{100.0};
    template <typename Scalar>
    Scalar HI{400.0};

    template <typename Scalar>
    GMM<Scalar>::GMM(int J, std::vector<Camera<Scalar>> camerasE, GMMParam<Scalar> gmmParamE, LBFGSpp::LBFGSParam<Scalar> lbfgsParamE) :
        J(J),
        cameras(std::move(camerasE)),
        gmmParam(std::move(gmmParamE)),
        lbfgsParam(std::move(lbfgsParamE)),
        frames({}),
        times(Vector<Scalar>::Zero(0)),
        hypothesisManager(cameras, gmmParam),
        gmmMaximizer(spline, designMatrix, hGrads, cameras, gmmParam, lbfgsParam),
        em(gmmParam.maxIter, gmmParam.tol, gmmMaximizer),
        hypothesisIds({})
    {
        designMatrix = RowMatrix<Scalar>::Zero(0, 0);
        spline = BSpline<Scalar>(gmmParam.splineDegree, UNIFORM),
        initGMMContainers();
    }

    template <typename Scalar>
    inline WorldPoints<Scalar> GMM<Scalar>::h(int i, const ColMatrix<Scalar> &theta)
    {
        return (designMatrix.row(i) * theta).transpose();
    }

    template <typename Scalar>
    void GMM<Scalar>::prepareGMMContainers(const std::map<int, ColMatrix<Scalar>> &initialThetas, const std::map<int, Vector<Scalar>> &initialPis)
    {
        for (int KEYPOINT : gmmParam.KEYPOINTS)
        {
            GMMContainer<Scalar> &gmmContainer = gmmContainers[KEYPOINT];

            auto foundTheta = initialThetas.find(KEYPOINT);
            if (foundTheta != initialThetas.end())
            {
                gmmContainer.parameters.theta = foundTheta->second;
            }
            else
            {
                if (!gmmParam.autoManageTheta)
                {
                    gmmContainer.parameters.theta = RowMatrix<double>::Random(spline.getNumBasis(), J * 3) * 50;
                }
            }

            #ifdef DEBUG_STATEMENTS
                std::cout << "Initial Theta for Keypoint: " << gmmContainer.KEYPOINT << std::endl;
                std::cout << gmmContainer.parameters.theta << std::endl;
            #endif

            auto foundPi = initialPis.find(KEYPOINT);
            if (foundPi != initialPis.end())
            {
                gmmContainer.parameters.pi = foundPi->second;
            }
            else
            {
                if (!gmmParam.autoManageTheta)
                {
                    gmmContainer.parameters.pi = Vector<Scalar>::Ones(J) / J;
                }
            }
        }
    }

    template <typename Scalar>
    void GMM<Scalar>::prepareDesignMatrix()
    {
        designMatrix = spline.designMatrix(times);
        spline.smoothDesignMatrix(designMatrix, gmmParam.splineSmoothingFactor);
        hGrads = {};
        int numBasis = spline.getNumBasis();

        for (int r = 0; r < designMatrix.rows(); r++)
        {
            Vector<Scalar> designMatrixRow = designMatrix.row(r);
            RowMatrix<Scalar> hGrad = RowMatrix<Scalar>::Zero(3, numBasis * 3);
            hGrad.row(0).head(numBasis) = designMatrixRow;
            hGrad.row(1).segment(numBasis, numBasis) = designMatrixRow;
            hGrad.row(2).tail(numBasis) = designMatrixRow;
            hGrads.push_back(hGrad);
        }
    }

    template <typename Scalar>
    inline void GMM<Scalar>::prepareCalculations(const std::map<int, ColMatrix<Scalar>> &initialThetas, const std::map<int, Vector<Scalar>> &initialPis)
    {
        prepareGMMContainers(initialThetas, initialPis);
        prepareDesignMatrix();
    }

    template <typename Scalar>
    std::map<int, EMFitResult<Scalar>> GMM<Scalar>::fit(const std::map<int, ColMatrix<Scalar>> &initialThetas, const std::map<int, Vector<Scalar>> &initialPis)
    {
        prepareCalculations(initialThetas, initialPis);
        std::map<int, EMFitResult<Scalar>> fitResults;
        EMFitResult<Scalar> fitResult;
        fitResult.supports = std::vector<bool>(J, false);

        std::vector<bool> hypothesisSupports(J, false);
        std::vector<int> nHypothesisSupports(J, 0);

        for (auto KEYPOINTIT = gmmParam.KEYPOINTS.begin(); KEYPOINTIT != gmmParam.KEYPOINTS.end(); ++KEYPOINTIT)
        {
            GMMContainer<Scalar> &gmmContainer = gmmContainers.find(*KEYPOINTIT)->second;
            fitResult.convergence = false;
            fitResult.diff = std::numeric_limits<Scalar>::infinity();
            fitResult.responsibilities = RowMatrix<Scalar>::Ones(gmmContainer.keyPoints.size(), J) / J;

            if (gmmContainer.keyPoints.size() == 0)
            {
                fitResult.parameters.theta = gmmContainer.parameters.theta;
                fitResult.parameters.pi = gmmContainer.parameters.pi;
                for (int j = 0; j < J; ++j)
                {
                    gmmContainer.supports[j].supported = false;
                    ++gmmContainer.supports[j].notSupportedSince;
                    fitResult.supports[j] = false;
                    ++nHypothesisSupports[j];
                }
            }
            else if (J == 0)
            {
                fitResult.parameters.theta = ColMatrix<Scalar>::Zero(spline.getNumBasis(), J * 3);
                fitResult.parameters.pi = Vector<Scalar>::Ones(J) / J;
            }
            else
            {

#ifdef DEBUG_STATEMENTS
                std::cout << "Starting with Thetas: " << gmmContainer.parameters.theta << std::endl;
#endif
                em(gmmContainer, fitResult);

                fitResult.parameters.theta = gmmContainer.parameters.theta;
                fitResult.parameters.pi = gmmContainer.parameters.pi;

#ifdef DEBUG_STATEMENTS
                std::cout << "No Supports Since: ";
#endif
                for (int j = 0; j < J; ++j)
                {
#ifdef DEBUG_STATEMENTS
                    std::cout << gmmContainer.supports[j].notSupportedSince << " ";
    #endif
                    
                    fitResult.supports[j] = gmmContainer.supports[j].supported;
                    hypothesisSupports[j] = hypothesisSupports[j] || gmmContainer.parameters.pi[j] > 1e-6 || gmmContainer.supports[j].notSupportedSince < gmmParam.notSupportedSinceThreshold;
                    nHypothesisSupports[j] += int(gmmContainer.supports[j].supported);
                }
#ifdef DEBUG_STATEMENTS
            std::cout << std::endl;
#endif
            }

            fitResults[*KEYPOINTIT] = fitResult;
        }

        int offset{int(designMatrix.rows()) - spline.getNumBasis() + 1};

        if (gmmParam.dragAlongUnsupportedKeyPoints)
        {
            int nSupportedMeanPoints = 0;
            ColMatrix<Scalar> newTheta(spline.getNumBasis(), 3);

            for (int j = 0; j < J; ++j)
            {
                newTheta.setOnes();
                nSupportedMeanPoints = 0;
                tempSupportedMeanPoint << 0, 0, 0;

                for (auto KEYPOINTIT = gmmParam.KEYPOINTS.begin(); KEYPOINTIT != gmmParam.KEYPOINTS.end(); ++KEYPOINTIT)
                {
                    GMMContainer<Scalar> &gmmContainer = gmmContainers[*KEYPOINTIT];

                    if (gmmContainer.supports[j].supported)
                    {
                        tempSupportedMeanPoint += h(offset, gmmContainer.parameters.theta.middleCols(j * 3, 3));
                        ++nSupportedMeanPoints;
                    }
                }

                if (nSupportedMeanPoints > 0)
                {
                    tempSupportedMeanPoint /= nSupportedMeanPoints;
                    newTheta.col(0) *= tempSupportedMeanPoint[0];
                    newTheta.col(1) *= tempSupportedMeanPoint[1];
                    newTheta.col(2) *= tempSupportedMeanPoint[2];

                    for (auto KEYPOINTIT = gmmParam.KEYPOINTS.begin(); KEYPOINTIT != gmmParam.KEYPOINTS.end(); ++KEYPOINTIT)
                    {
                        GMMContainer<Scalar> &gmmContainer = gmmContainers.find(*KEYPOINTIT)->second;

                        if (!gmmContainer.supports[j].supported)
                        {
                            gmmContainer.parameters.theta.middleCols(j * 3, 3) = newTheta;
                        }
                    }
                }
            }
        }

        for (int i = hypothesisSupports.size()-1; i >= 0; --i)
        {
            if (!hypothesisSupports[i])
            {
                removeHypothesis(i);
                removeHypothesisKeyPoints(i, fitResults);
            }
        }

        return fitResults;
    }

    template <typename Scalar>
    inline void GMM<Scalar>::addHypothesis()
    {
        Scalar r{LO<Scalar> + static_cast<Scalar>(rand()) / (static_cast<Scalar>(RAND_MAX / (HI<Scalar> - LO<Scalar>)))};
        addHypothesis(WorldPoint<Scalar>::Ones(3) * r);
    }

    template <typename Scalar>
    void GMM<Scalar>::addHypothesis(const WorldPoint<Scalar> &worldPoint)
    {
        if (gmmParam.autoManageTheta)
        {
            ColMatrix<Scalar> newTheta = ColMatrix<Scalar>::Ones(spline.getNumBasis(), 3);
            newTheta.col(0) *= worldPoint[0];
            newTheta.col(1) *= worldPoint[1];
            newTheta.col(2) *= worldPoint[2];

#ifdef DEBUG_STATEMENTS
            std::cout << "New Theta: " << newTheta << std::endl;
        #endif

            for (auto KEYPOINT : gmmParam.KEYPOINTS)
            {
                GMMContainer<Scalar> &gmmContainer = gmmContainers[KEYPOINT];
                gmmContainer.parameters.theta.conservativeResize(Eigen::NoChange, (J + 1) * 3);
                gmmContainer.parameters.pi.conservativeResize(J + 1);

                gmmContainer.parameters.theta.rightCols(3) = newTheta;

                if (gmmContainer.keyPoints.size() == 0)
                {
                    gmmContainer.parameters.pi.setOnes();
                    gmmContainer.parameters.pi /= Scalar(J + 1);
                }
                else
                {
                    Scalar newPi{1.0 / Scalar(gmmContainer.keyPoints.size())};
                    Scalar piToRemove{newPi / Scalar(J)};

                    for (int j = 0; j < J; ++j)
                    {
                        gmmContainer.parameters.pi[j] -= piToRemove;
                    }
                    gmmContainer.parameters.pi[J] = newPi;
                }

                gmmContainer.hypothesisChanged = true;
            }

            hypothesisIds.push_back(currentHypothesisId++);
        }

        for (auto KEYPOINT : gmmParam.KEYPOINTS)
        {
            gmmContainers[KEYPOINT].supports.push_back(Support(frames.size()-1));
        }
        ++J;
#ifdef DEBUG_STATEMENTS
        std::cout << "Adding Hypothesis to J = " << J << std::endl;
#endif
    }

    template <typename Scalar>
    void GMM<Scalar>::removeHypothesis(int index)
    {
        if (gmmParam.autoManageTheta)
        {
            for (auto KEYPOINT : gmmParam.KEYPOINTS)
            {
                GMMContainer<Scalar> &gmmContainer = gmmContainers[KEYPOINT];

                if (index != J - 1)
                {
                    gmmContainer.parameters.theta.middleCols(index * 3, (J - index - 1) * 3) = gmmContainer.parameters.theta.rightCols((J - index - 1) * 3);
                    gmmContainer.parameters.pi.segment(index, J - index - 1) = gmmContainer.parameters.pi.tail(J - index - 1);
                }

                Scalar piToAdd = gmmContainer.parameters.pi[index] / Scalar(J - 1);

                for (int j = 0; j < J - 1; ++j)
                {
                    gmmContainer.parameters.pi[j] += piToAdd;
                }

                gmmContainer.parameters.theta.conservativeResize(Eigen::NoChange, (J - 1) * 3);
                gmmContainer.parameters.pi.conservativeResize(J - 1);

                gmmContainer.hypothesisChanged = true;
            }

            hypothesisIds.erase(hypothesisIds.begin() + index);
        }

        hypothesisManager.removeHypothesis(index);
        J--;
#ifdef DEBUG_STATEMENTS
        std::cout << "Removing Hypothesis to J = " << J << std::endl;
#endif
    }

    template <typename Scalar>
    void GMM<Scalar>::removeHypothesisKeyPoints(int index, std::map<int, EMFitResult<Scalar>> &fitResults)
    {
        for (auto KEYPOINT : gmmParam.KEYPOINTS)
        {
            gmmContainers[KEYPOINT].supports.erase(gmmContainers[KEYPOINT].supports.begin() + index);

            for (int r = fitResults[KEYPOINT].responsibilities.rows() - 1; r >= 0; --r)
            {
                int index_of_max = -1;
                int index_of_max2 = -1;
                fitResults[KEYPOINT].responsibilities.row(r).maxCoeff(&index_of_max, &index_of_max2);

                if (index_of_max2 == index)
                {
                    gmmContainers[KEYPOINT].keyPoints.erase(gmmContainers[KEYPOINT].keyPoints.begin() + r);
                }
            }
        }
    }

    template <typename Scalar>
    void GMM<Scalar>::addFrame(Frame<Scalar> &frame)
    {
        if (gmmParam.minValidKeyPoints > 0)
        {
            filterPeople(frame);
        }
        addFrameToGMMContainers(frame);
        addTimeToBSpline(frame.time);

        if (gmmParam.autoManageTheta)
        {
            fillupThetas();
        }

        if (frames.size() > gmmParam.maxFrameBuffer)
        {
            if (spline.getKnots()[spline.getDegree() + 1] < frames[0].time)
            {
                spline.popKnotFront();

                if (gmmParam.autoManageTheta)
                {
                    for (auto KEYPOINT : gmmParam.KEYPOINTS)
                    {
#ifdef DEBUG_STATEMENTS
                        std::cout << "Thetas Before Frame Drop:" << gmmContainers[KEYPOINT].parameters.theta << std::endl;
#endif

                        GMMContainer<Scalar>& gmmContainer = gmmContainers[KEYPOINT];
                        gmmContainer.parameters.theta.topRows(spline.getNumBasis()) = gmmContainer.parameters.theta.bottomRows(spline.getNumBasis());
                        gmmContainer.parameters.theta.conservativeResize(spline.getNumBasis(), Eigen::NoChange);

#ifdef DEBUG_STATEMENTS
                        std::cout << "Thetas After Frame Drop:" << gmmContainers[KEYPOINT].parameters.theta << std::endl;
#endif
                    }
                }
            }

            for (auto keypoint : gmmParam.KEYPOINTS)
            {
                gmmContainers[keypoint].dropFrame();
            }

            frames.erase(frames.begin());
        }

        if (gmmParam.autoManageHypothesis)
        {
            std::vector<CameraPoint<Scalar>> newHypothesisPoints;
            std::vector<size_t> existingHypothesisIndizes;
            hypothesisManager.update(frame, newHypothesisPoints, existingHypothesisIndizes);

            for (const CameraPoint<Scalar> &newHypothesisPoint : newHypothesisPoints)
            {
                WorldPoint<Scalar> newPIs = cameras[frame.cameraIndex].pixelsToWorldPoints(newHypothesisPoint).col(0);
                addHypothesis(newPIs);
            }
        }
    }

    template <typename Scalar>
    void GMM<Scalar>::filterPeople(Frame<Scalar> &frame)
    {
        if (frame.kpts.size() > 0)
        {
            for (int p = frame.kpts.size() - 1; p >= 0; --p)
            {   
                int nValidKeyPoints = 0;
                for (auto KEYPOINT : gmmParam.KEYPOINTS)
                {
                    /*if (!cameras[frame.cameraIndex].isPointInFrame(frame.kpts[p].row(KEYPOINT).head(2)))
                    {
                        frame.kpts[p](KEYPOINT, 2) = 0;
                    }*/

                    if (frame.kpts[p](KEYPOINT, 2) >= gmmParam.keypointConfidenceThreshold)
                    {
                        ++nValidKeyPoints;
                    }
                }

                if (nValidKeyPoints <= gmmParam.minValidKeyPoints)
                {
                    frame.kpts.erase(frame.kpts.begin() + p);
                }
            }
        }
    }

    template <typename Scalar>
    GMM<Scalar> &operator<<(GMM<Scalar> &self, Frame<Scalar> &frame)
    {
        self.addFrame(frame);
        return self;
    }

    template <typename Scalar>
    void GMM<Scalar>::addTimeToBSpline(Scalar time)
    {
        if (spline.getKnots().size() == 0)
        {
            spline.setKnots((Vector<Scalar>(2) << time, time + gmmParam.splineKnotDelta).finished());
        }

        Scalar lastKnot = spline.getKnots()[spline.getKnots().size() - (1 + spline.getDegree())];

        if (lastKnot < time)
        {
            spline.pushKnot(lastKnot + gmmParam.splineKnotDelta);
        }
    }

    template <typename Scalar>
    void GMM<Scalar>::fillupThetas()
    {
        for (auto KEYPOINT : gmmParam.KEYPOINTS)
        {
            GMMContainer<Scalar> &gmmContainer = gmmContainers[KEYPOINT];
            int oldNumberOfRows = gmmContainer.parameters.theta.rows();

            if (oldNumberOfRows < spline.getNumBasis())
            {
                ColMatrix<Scalar> rowsToAdd;

                if (gmmParam.copyLastThetas && gmmContainer.parameters.theta.rows() >= spline.getNumBasis() - oldNumberOfRows)
                {
                    rowsToAdd = gmmContainer.parameters.theta.bottomRows(spline.getNumBasis() - oldNumberOfRows);
                }
                else
                {
                    rowsToAdd = ColMatrix<Scalar>::Random(spline.getNumBasis() - oldNumberOfRows, J * 3) * 50;
                }

                gmmContainer.parameters.theta.conservativeResize(spline.getNumBasis(), Eigen::NoChange);
                gmmContainer.parameters.theta.bottomRows(spline.getNumBasis() - oldNumberOfRows) = rowsToAdd;
            }
        }
    }

    template <typename Scalar>
    void GMM<Scalar>::addFrameToGMMContainers(const Frame<Scalar> &frame)
    {
        frames.push_back(frame);

        if (times.size() == gmmParam.maxFrameBuffer)
        {
            times.head(times.size() - 1) = times.tail(times.size() - 1);
        }
        else
        {
            times.conservativeResize(times.size() + 1);
        }

        times(times.size() - 1) = frame.time;

        if (frame.kpts.size() > 0)
        {
            for (RowMatrix<Scalar> person : frame.kpts)
            {
                for (auto KEYPOINT : gmmParam.KEYPOINTS)
                {
                    if (person(KEYPOINT, 2) >= gmmParam.keypointConfidenceThreshold)
                    {
                        #ifdef DEBUG_STATEMENTS
                            std::cout << "Keypoint to Add: " << person.row(KEYPOINT) << std::endl;
                        #endif

                        gmmContainers[KEYPOINT].addKeypoint(person.row(KEYPOINT).head(2), frame.time, frame.cameraIndex, frames.size() - 1);
                    }
                }
            }
        }
    }

    template <typename Scalar>
    void GMM<Scalar>::initGMMContainers()
    {
        for (int KEYPOINT : gmmParam.KEYPOINTS)
        {
            gmmContainers.insert(std::make_pair(KEYPOINT, GMMContainer<Scalar>(KEYPOINT, J, cameras, gmmParam.nu, designMatrix)));
            gmmContainers[KEYPOINT].parameters.pi = Vector<Scalar>::Ones(J) / J;
        }
    }
};
#endif