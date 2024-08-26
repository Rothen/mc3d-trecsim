#ifndef BSPLINE_H
#define BSPLINE_H

#include "config.h"
#include "mc3d_common.h"

#include <iostream>
#include <optional>
#include <vector>
#include <Eigen/Dense>

namespace MC3D_TRECSIM
{
    enum AUGMENTATION_MODE { UNIFORM, SAME, NONE };

    template <typename Scalar>
    class BSpline
    {
    public:
        BSpline(unsigned int degree = 0, AUGMENTATION_MODE augmentationMode = NONE, Vector<Scalar> knots = Vector<Scalar>::Zero(0));

        void setKnots(Vector<Scalar> knots);

        const Vector<Scalar> &getKnots() const;

        int getNumBasis() const;

        int getDegree() const;

        Scalar basis(Scalar t, int j, int k);

        Scalar basisGrad(Scalar t, int j, int k);

        Scalar basisInt(const int j, const int k);

        RowMatrix<Scalar> designMatrix(const Vector<Scalar> &t);

        void smoothDesignMatrix(RowMatrix<Scalar> &designMatrix, const Scalar lambda);

        void designMatrix(const Vector<Scalar> &t, RowMatrix<Scalar> &A);

        void pushKnot(Scalar knot);

        void popKnotFront();

    private:
        Vector<Scalar> knots;
        int degree;
        int numBasis;
        AUGMENTATION_MODE augmentationMode;
    };
}
#include "bspline_impl.h"
#endif