#ifndef MULTIVARIATE_NORMAL_H
#define MULTIVARIATE_NORMAL_H

#include "config.h"
#include "mc3d_common.h"

#include <Eigen/Dense>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MC3D_TRECSIM
{
    template <typename Scalar, int nComponents>
    class MultivariateNormal
    {
    public:
        using NVariatePoint = Eigen::Vector<Scalar, nComponents>;
        using NVariateCovariance = Eigen::Matrix<Scalar, nComponents, nComponents, Eigen::RowMajor>;

        MultivariateNormal(NVariatePoint mean = NVariatePoint::Zero(nComponents), NVariateCovariance covariance = NVariateCovariance::Identity(nComponents, nComponents));

        inline void setMean(NVariatePoint mean);

        inline void setCovariance(NVariateCovariance covariance);

        inline const NVariatePoint& getMean() const;

        inline const NVariateCovariance& getCovariance() const;

        inline void calculateCovarianceFactors();

        inline Scalar pdf(const NVariatePoint& x) const;

        inline Scalar logPdf(const NVariatePoint& x) const;

        inline Scalar pdf(const NVariatePoint &x, const NVariatePoint &mean) const;

        inline Scalar logPdf(const NVariatePoint &x, const NVariatePoint &mean) const;

    private:
        NVariatePoint mean;
        NVariateCovariance covariance;
        NVariateCovariance covarianceInf;
        Scalar det;
        Scalar k;
        Scalar factor;
        Scalar logFactor;
    };
}
#include "multivariate_normal_impl.h"
#endif