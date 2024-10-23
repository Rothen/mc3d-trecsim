#ifndef MULTIVARIATE_NORMAL_IMPL_H
#define MULTIVARIATE_NORMAL_IMPL_H

#include "multivariate_normal.h"

namespace mc3d
{
    template <int nComponents>
    MultivariateNormal<nComponents>::MultivariateNormal(Tensor mean, Tensor covariance) : 
        mean(mean),
        covariance(covariance)
    {
        calculate_covariance_factors();
    }

    template <int nComponents>
    inline void MultivariateNormal<nComponents>::set_mean(Tensor mean)
    {
        this->mean = std::move(mean);
    }

    template <int nComponents>
    inline void MultivariateNormal<nComponents>::set_covariance(Tensor covariance)
    {
        this->covariance = std::move(covariance);
        calculate_covariance_factors();
    }

    template <int nComponents>
    inline const Tensor MultivariateNormal<nComponents>::get_mean() const
    {
        return mean;
    }

    template <int nComponents>
    inline const Tensor MultivariateNormal<nComponents>::get_covariance() const
    {
        return covariance;
    }

    template <int nComponents>
    inline void MultivariateNormal<nComponents>::calculate_covariance_factors()
    {
        torch::Tensor Q, R;
        std::tie(Q, R) = torch::linalg_qr(covariance);

        covariance_inf = -0.5 * R.inverse().mm(Q.transpose(0, 1));
        k = torch::tensor({RealType(nComponents)});
        det = covariance.det();
        factor = det.pow(-0.5) * pow(2 * M_PI, -static_cast<RealType>(nComponents) / 2);
        log_factor = factor.log();
    }

    template <int nComponents>
    inline Tensor MultivariateNormal<nComponents>::pdf(const Tensor x) const
    {
        return pdf(x, mean);
    }

    template <int nComponents>
    inline Tensor MultivariateNormal<nComponents>::log_pdf(const Tensor x) const
    {
        return log_pdf(x, mean);
    }

    template <int nComponents>
    inline Tensor MultivariateNormal<nComponents>::pdf(const Tensor x, const Tensor mean) const
    {
        return factor.mm((x - mean).transpose(0, 1).mm(covariance_inf).mm(x - mean).exp());
    }

    template <int nComponents>
    inline Tensor MultivariateNormal<nComponents>::log_pdf(const Tensor x, const Tensor mean) const
    {
        return log_factor + ((x - mean).transpose(0, 1).mm(covariance_inf).mm(x - mean));
    }
}
#endif