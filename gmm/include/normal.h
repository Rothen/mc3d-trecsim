#ifndef NORMAL_H
#define NORMAL_H

#include <math.h>

#include "mc3d_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mc3d
{
    class Normal
    {
    public:
        Normal(RealType mean = 0, RealType variance = 1);

        inline void set_mean(RealType mean);

        inline void set_variance(RealType variance);

        inline const RealType get_mean() const;

        inline const RealType get_variance() const;

        inline void calculate_variance_factors();

        inline RealType prob(const RealType x) const;

        inline RealType log_prob(const RealType x) const;

        inline RealType prob(const RealType x, const RealType mean) const;

        inline RealType log_prob(const RealType x, const RealType mean) const;

    private:
        RealType mean;
        RealType variance;
        RealType variance_inf;
        RealType det;
        RealType k;
        RealType factor;
        RealType log_factor;
    };
}

#include "normal_impl.h"

#endif