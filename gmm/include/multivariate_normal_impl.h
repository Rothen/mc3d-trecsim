#ifndef MULTIVARIATE_NORMAL_IMPL_H
#define MULTIVARIATE_NORMAL_IMPL_H

#include "multivariate_normal.h"

namespace mc3d
{
    MultivariateNormal::MultivariateNormal(Tensor mean, Tensor covariance) : 
        mean(mean),
        covariance(covariance)
    {
        calculate_covariance_factors();
    }

    inline void MultivariateNormal::set_mean(Tensor mean)
    {
        this->mean = mean;
    }

    inline void MultivariateNormal::set_covariance(Tensor covariance)
    {
        this->covariance = covariance;
        calculate_covariance_factors();
    }

    inline const Tensor MultivariateNormal::get_mean() const
    {
        return mean;
    }

    inline const Tensor MultivariateNormal::get_covariance() const
    {
        return covariance;
    }

    inline void MultivariateNormal::calculate_covariance_factors()
    {
        covariance_inf = -0.5 * covariance.inverse();
        k = static_cast<RealType>(mean.size(0));
        det = covariance.det();
        factor = det.pow(-0.5) * pow(2 * M_PI, -k / 2);
        log_factor = factor.log();
    }

    inline Tensor MultivariateNormal::prob(const Tensor x) const
    {
        return prob(x, mean);
    }

    inline Tensor MultivariateNormal::log_prob(const Tensor x) const
    {
        return log_prob(x, mean);
    }

    inline Tensor MultivariateNormal::prob(const Tensor x, const Tensor mean) const
    {
        return factor * (x - mean).transpose(0, 1).mm(covariance_inf).mm(x - mean).exp();
    }

    inline Tensor MultivariateNormal::log_prob(const Tensor x, const Tensor mean) const
    {
        return log_factor + ((x - mean).transpose(0, 1).mm(covariance_inf).mm(x - mean));
    }
}
#endif