#ifndef NORMAL_IMPL_H
#define NORMAL_IMPL_H

#include "normal.h"

namespace mc3d
{
    Normal::Normal(RealType mean, RealType variance) : 
        mean(mean),
        variance(variance)
    {
        calculate_variance_factors();
    }

    inline void Normal::set_mean(RealType mean)
    {
        this->mean = mean;
    }

    inline void Normal::set_variance(RealType variance)
    {
        this->variance = variance;
        calculate_variance_factors();
    }

    inline const RealType Normal::get_mean() const
    {
        return mean;
    }

    inline const RealType Normal::get_variance() const
    {
        return variance;
    }

    inline void Normal::calculate_variance_factors()
    {
        variance_inf = -0.5 * variance.inverse();
        k = static_cast<RealType>(mean.size(0));
        det = variance.det();
        factor = det.pow(-0.5) * pow(2 * M_PI, -k / 2);
        log_factor = factor.log();
    }

    inline RealType Normal::prob(const RealType x) const
    {
        return prob(x, mean);
    }

    inline RealType Normal::log_prob(const RealType x) const
    {
        return log_prob(x, mean);
    }

    inline RealType Normal::prob(const RealType x, const RealType mean) const
    {
        return 1 / (std::sqrt(2 * M_PI * sigma_squared)) * (-std::pow(x - mean, 2) / (2 * sigma_squared)).exp();
    }

    inline RealType Normal::log_prob(const RealType x, const RealType mean) const
    {
        return -1 / 2 * ();
    }
}
#endif