#ifndef POSE_ID_MEASUREMENT_IMPL_H
#define POSE_ID_MEASUREMENT_IMPL_H

#include "pose_id_measurement.h"

namespace mc3d
{
    PoseIdMeasurement::PoseIdMeasurement(RealType prec_variance) : log_normalization{log(2 * pi * prec_variance)}
    { }

    Tensor PoseIdMeasurement::pose_id_log_likelihood(const PersonHypothesis &hypothesis) const
    {
        using torch::indexing::Slice;
        Tensor sum{torch::zeros({1})}; // use global shared torch::TensorOptions() object to define type, grad,...
        static constexpr RealType half{RealType(1) / RealType(2)};
        for (auto &measurement : measurements)
        {
            for (size_t m_ind{0}, end{measurement.limb_ids.size()}; m_ind < end; ++m_ind)
            {
                Tensor tmp{measurement.values.index({static_cast<int64_t>(m_ind), Slice()}) - hypothesis.predict(measurement.limb_ids[m_ind], measurement.time, measurement.camera)};
                sum -= half * torch::inner(tmp, tmp) + log_normalization;
            }
        }
        return sum;
    }
}

#endif