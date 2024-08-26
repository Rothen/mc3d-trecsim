#ifndef BSPLINE_IMPL_H
#define BSPLINE_IMPL_H

#include "bspline.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <optional>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
#include <map>
#include <tuple>

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    BSpline<Scalar>::BSpline(unsigned int degree, AUGMENTATION_MODE augmentationMode, Vector<Scalar> knots) :
        degree(degree),
        augmentationMode(augmentationMode),
        numBasis(0)
    {
        if (knots.size() != 0)
        {
            setKnots(std::move(knots));
        }
    }

    template <typename Scalar>
    void BSpline<Scalar>::setKnots(Vector<Scalar> knots)
    {
        switch (augmentationMode)
        {
        case UNIFORM:
        {
            Scalar diffBegin = abs(knots[0] - knots[1]);
            Scalar diffEnd = abs(knots[knots.size() - 2] - knots[knots.size() - 1]);

            this->knots = Vector<Scalar>::Zero(knots.size() + degree * 2);
            this->knots.segment(degree, knots.size()) = knots;

            for (int i = 0; i < degree; i++)
            {
                this->knots[degree - i - 1] = this->knots[degree - i] - diffBegin;
                this->knots[this->knots.size() - degree + i] = this->knots[this->knots.size() - degree + i - 1] + diffEnd;
            }
            break;
        }
        case SAME:
        {
            this->knots = Vector<Scalar>::Zero(knots.size() + degree * 2);
            this->knots.segment(degree, knots.size()) = knots;

            for (int i = 0; i < degree; i++)
            {
                this->knots[i] = this->knots[degree];
                this->knots[this->knots.size() - 1 - i] = this->knots[this->knots.size() - 1 - degree];
            }
            break;
        }
        case NONE:
        {
            this->knots = std::move(knots);
            break;
        }
        }

        numBasis = this->knots.size() - degree - 1;
    }

    template <typename Scalar>
    const Vector<Scalar>& BSpline<Scalar>::getKnots() const
    {
        return knots;
    }

    template <typename Scalar>
    int BSpline<Scalar>::getNumBasis() const
    {
        return numBasis;
    }

    template <typename Scalar>
    int BSpline<Scalar>::getDegree() const
    {
        return degree;
    }

    template <typename Scalar>
    Scalar BSpline<Scalar>::basis(Scalar t, int i, int k)
    {
        if (k == 0)
        {
            return Scalar(knots[i] <= t && t < knots[i + 1]);
        }

        double denom1 = knots[i + k] - knots[i];
        double denom2 = knots[i + k + 1] - knots[i + 1];

        double term1 = (denom1 != 0) ? (t - knots[i]) / denom1 * basis(t, i, k - 1) : 0;
        double term2 = (denom2 != 0) ? (knots[i + k + 1] - t) / denom2 * basis(t, i + 1, k - 1) : 0;

        return term1 + term2;
    }

    template <typename Scalar>
    Scalar BSpline<Scalar>:: basisGrad(Scalar t, int j, int k)
    {
        if (k == 0)
        {
            return Scalar(t >= knots[j] && t < knots[j + 1]);
        }

        Scalar leftDenom{knots[j + k] - knots[j]};
        Scalar rightDenom{knots[j + k + 1] - knots[j + 1]};

        Scalar left{0.0};
        Scalar right{0.0};

        if (leftDenom != 0.0)
        {
            left = (k * basisGrad(t, j, k - 1)) / leftDenom;
        }

        if (rightDenom != 0.0)
        {
            right = (k * basisGrad(t, j + 1, k - 1)) / rightDenom;
        }

        return left - right;
    }

    template <typename Scalar>
    Scalar BSpline<Scalar>::basisInt(const int j, const int k)
    {
        /*Scalar h{(knots[knots.size() - 1] - knots[0]) / 100};
        Scalar integral{0.5 * (basis(knots[0], j, k) + basis(knots[knots.size() - 1], j, k))};
        for (int i = 1; i < 100; ++i)
        {
            integral += basis(knots[0] + i * h, j, k);
        }
        integral *= h;
        return integral;*/
        return (knots[j + k + 1] - knots[j]) / (k + 1);
    }

    template <typename Scalar>
    RowMatrix<Scalar> BSpline<Scalar>::designMatrix(const Vector<Scalar> &t)
    {
        RowMatrix<Scalar> r_designMatrix = RowMatrix<Scalar>::Zero(t.size(), numBasis);

        for (int j = 0; j < t.size(); j++)
        {
            for (int i = 0; i < numBasis; i++)
            {
                r_designMatrix(j, i) = basis(t[j], i, degree);
            }
        }

        return r_designMatrix;
    }

    template <typename Scalar>
    void BSpline<Scalar>::smoothDesignMatrix(RowMatrix<Scalar> &designMatrix, const Scalar lambda)
    {
        const int offset = designMatrix.rows();
        designMatrix.conservativeResize(designMatrix.rows() + numBasis - 2, Eigen::NoChange);
        designMatrix.bottomRows(numBasis - 2) = RowMatrix<Scalar>::Zero(numBasis - 2, numBasis);

        Scalar N;
        Scalar s;
        Scalar d_j;
        Scalar b0;
        Scalar b2;
        Scalar b1;

        for (int j = 2; j < numBasis; j++)
        {
            auto basisFunc = [&](Scalar t)
            {
                return basis(t, j, degree-2);
            };
            N = basisInt(j, degree - 2);
            s = std::sqrt(N);
            d_j = (degree - 1) * (degree - 2) * s / (knots[j + degree - 2] - knots[j]);
            b0 = d_j / (knots[j + degree - 2] - knots[j - 1]);
            b2 = d_j / (knots[j + degree - 1] - knots[j]);
            b1 = -(b0 + b2);
            designMatrix.row(offset + j - 2).segment(j - 2, 3) = lambda * (Vector<Scalar>(3) << b0, b1, b2).finished();
        }
    }

    template <typename Scalar>
    void BSpline<Scalar>::designMatrix(const Vector<Scalar> &t, RowMatrix<Scalar> &A)
    {
        for (int i = 0; i < numBasis; i++)
        {
            for (int j = 0; j < t.size(); j++)
            {
                A(j, i) = basis(t[j], i, degree);
            }
        }
    }

    template <typename Scalar>
    void BSpline<Scalar>::pushKnot(Scalar knot)
    {
        knots.conservativeResize(knots.size() + 1);

        switch (augmentationMode)
        {
        case UNIFORM:
        {
            knots[knots.size() - degree - 1] = knot;
            Scalar diffEnd = abs(knots[knots.size() - degree - 2] - knots[knots.size() - degree - 1]);

            for (int i = 0; i < degree; i++)
            {
                knots[knots.size() - degree + i] = knots[knots.size() - degree + i - 1] + diffEnd;
            }
            break;
        }
        case SAME:
        {
            for (int i = 0; i < degree + 1; i++)
            {
                knots[knots.size() - i - 1] = knot;
            }
            break;
        }
        case NONE:
        {
            knots[knots.size() - 1] = knot;
            break;
        }
        }

        numBasis = knots.size() - degree - 1;
    }

    template <typename Scalar>
    void BSpline<Scalar>::popKnotFront()
    {
        knots.head(knots.size() - 1) = knots.tail(knots.size() - 1);
        knots.conservativeResize(knots.size() - 1);

        switch (augmentationMode)
        {
        case UNIFORM:
        {
            Scalar diffBegin = abs(knots[degree] - knots[degree + 1]);

            for (int i = 0; i < degree; i++)
            {
                knots[degree - i - 1] = knots[degree - i] - diffBegin;
            }
            break;
        }
        case SAME:
        {
            for (int i = 0; i < degree; i++)
            {
                knots[i] = knots[degree];
            }
            break;
        }
        case NONE:
        {
            break;
        }
        }

        numBasis = knots.size() - degree - 1;
    }
}
#endif