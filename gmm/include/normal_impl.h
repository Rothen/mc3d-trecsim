#ifndef NORMAL_IMPL_H
#define NORMAL_IMPL_H

#include "normal.h"

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    Normal<Scalar>::Normal(Scalar mean, Scalar variance) : mean(std::move(mean)), variance(std::move(variance))
    {
        calculateVarianceFactors();
    }

    template <typename Scalar>
    inline void Normal<Scalar>::setMean(Scalar mean)
    {
        this->mean = std::move(mean);
    }

    template <typename Scalar>
    inline void Normal<Scalar>::setVariance(Scalar variance)
    {
        this->variance = std::move(variance);
        calculateVarianceFactors();
    }

    template <typename Scalar>
    inline const Scalar &Normal<Scalar>::getMean() const
    {
        return mean;
    }

    template <typename Scalar>
    inline const Scalar &Normal<Scalar>::getVariance() const
    {
        return variance;
    }

    template <typename Scalar>
    inline void Normal<Scalar>::calculateVarianceFactors()
    {
        sigma = sqrt(variance);
        factor = 1.0 / sqrt(2 * M_PI * variance);
        logFactor = -std::log(sqrt(2 * M_PI * variance));
        varianceFactor = 2 * variance;
    }

    template <typename Scalar>
    inline Scalar Normal<Scalar>::pdf(const Scalar &x) const
    {
        return pdf(x, mean);
    }

    template <typename Scalar>
    inline Scalar Normal<Scalar>::logPdf(const Scalar &x) const
    {
        return logPdf(x, mean);
    }

    template <typename Scalar>
    inline Scalar Normal<Scalar>::pdf(const Scalar &x, const Scalar &mean) const
    {
        return factor * exp(- (std::pow(x - mean, 2) / varianceFactor));
    }

    template <typename Scalar>
    inline Scalar Normal<Scalar>::logPdf(const Scalar &x, const Scalar &mean) const
    {
        return logFactor - (std::pow(x - mean, 2) / varianceFactor);
    }
}
#endif