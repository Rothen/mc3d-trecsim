#ifndef MULTIVARIATE_NORMAL_H
#define MULTIVARIATE_NORMAL_H

#include <math.h>

#include "mc3d_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mc3d
{
    class MultivariateNormal
    {
    public:
        MultivariateNormal(Tensor mean = torch::zeros({1, 1}), Tensor covariance = torch::eye(1));

        inline void set_mean(Tensor mean);

        inline void set_covariance(Tensor covariance);

        inline const Tensor get_mean() const;

        inline const Tensor get_covariance() const;

        inline void calculate_covariance_factors();

        inline Tensor pdf(const Tensor x) const;

        inline Tensor log_pdf(const Tensor x) const;

        inline Tensor pdf(const Tensor x, const Tensor mean) const;

        inline Tensor log_pdf(const Tensor x, const Tensor mean) const;

    private:
        Tensor mean;
        Tensor covariance;
        Tensor covariance_inf;
        Tensor det;
        RealType k;
        Tensor factor;
        Tensor log_factor;
    };
}

#include "multivariate_normal_impl.h"

#endif