#ifndef NORMAL_H
#define NORMAL_H

#include "config.h"
#include "mc3d_common.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MC3D_TRECSIM
{
    template <typename Scalar>
    class Normal
    {
    public:
        Normal(Scalar mean = 0.0, Scalar variance = 1.0);

        inline void setMean(Scalar mean);

        inline void setVariance(Scalar variance);

        inline const Scalar &getMean() const;

        inline const Scalar &getVariance() const;

        inline void calculateVarianceFactors();

        inline Scalar pdf(const Scalar &x) const;

        inline Scalar logPdf(const Scalar &x) const;

        inline Scalar pdf(const Scalar &x, const Scalar &mean) const;

        inline Scalar logPdf(const Scalar &x, const Scalar &mean) const;

    private:
        Scalar mean;
        Scalar variance;
        Scalar sigma;

        Scalar factor;
        Scalar logFactor;
        Scalar varianceFactor;
    };
}
#include "normal_impl.h"
#endif