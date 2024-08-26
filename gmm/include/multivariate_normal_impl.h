#ifndef MULTIVARIATE_NORMAL_IMPL_H
#define MULTIVARIATE_NORMAL_IMPL_H

#include "multivariate_normal.h"

namespace MC3D_TRECSIM
{
    template <typename Scalar, int nComponents>
    using NVariatePoint = Eigen::Vector<Scalar, nComponents>;
    template <typename Scalar, int nComponents>
    using NVariateCovariance = Eigen::Matrix<Scalar, nComponents, nComponents, Eigen::RowMajor>;

    template <typename Scalar, int nComponents>
    MultivariateNormal<Scalar, nComponents>::MultivariateNormal(NVariatePoint mean, NVariateCovariance covariance) : 
        mean(std::move(mean)),
        covariance(std::move(covariance))
    {
        calculateCovarianceFactors();
    }

    template <typename Scalar, int nComponents>
    inline void MultivariateNormal<Scalar, nComponents>::setMean(NVariatePoint mean)
    {
        this->mean = std::move(mean);
    }

    template <typename Scalar, int nComponents>
    inline void MultivariateNormal<Scalar, nComponents>::setCovariance(NVariateCovariance covariance)
    {
        this->covariance = std::move(covariance);
        calculateCovarianceFactors();
    }

    template <typename Scalar, int nComponents>
    inline const Eigen::Vector<Scalar, nComponents> &MultivariateNormal<Scalar, nComponents>::getMean() const
    {
        return mean;
    }

    template <typename Scalar, int nComponents>
    inline const Eigen::Matrix<Scalar, nComponents, nComponents, Eigen::RowMajor> &MultivariateNormal<Scalar, nComponents>::getCovariance() const
    {
        return covariance;
    }

    template <typename Scalar, int nComponents>
    inline void MultivariateNormal<Scalar, nComponents>::calculateCovarianceFactors()
    {
        Eigen::ColPivHouseholderQR<NVariateCovariance> decomp(covariance);
        covarianceInf = -0.5 * decomp.matrixR().inverse() * decomp.matrixQ().transpose();
        k = decomp.rank();
        det = covariance.determinant();
        factor = pow(2 * M_PI, -k / 2) * pow(det, -0.5);
        logFactor = std::log(factor);
    }

    template <typename Scalar, int nComponents>
    inline Scalar MultivariateNormal<Scalar, nComponents>::pdf(const NVariatePoint &x) const
    {
        return pdf(x, mean);
    }

    template <typename Scalar, int nComponents>
    inline Scalar MultivariateNormal<Scalar, nComponents>::logPdf(const NVariatePoint &x) const
    {
        return logPdf(x, mean);
    }

    template <typename Scalar, int nComponents>
    inline Scalar MultivariateNormal<Scalar, nComponents>::pdf(const NVariatePoint &x, const NVariatePoint &mean) const
    {
        return factor * exp((x - mean).transpose() * covarianceInf * (x - mean));
    }

    template <typename Scalar, int nComponents>
    inline Scalar MultivariateNormal<Scalar, nComponents>::logPdf(const NVariatePoint &x, const NVariatePoint &mean) const
    {
        return logFactor + ((x - mean).transpose() * covarianceInf * (x - mean));
    }
}
#endif