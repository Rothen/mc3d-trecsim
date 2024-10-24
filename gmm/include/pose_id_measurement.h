#ifndef POSE_ID_MEASUREMENT_H
#define POSE_ID_MEASUREMENT_H

#include <numbers>

#include "mc3d_common.h"
#include "person_hypothesis.h"
#include "pose_measurement.h"

using namespace std::numbers;

namespace mc3d
{
    class PoseIdMeasurement
    {
    public:
        PoseIdMeasurement(RealType prec_variance);
        // clear,add measurement, ...

        Tensor pose_id_log_likelihood(const PersonHypothesis &hypothesis) const;

    protected:
        RealType log_normalization;
        vector<PoseMeasurement> measurements;
    };
}

#include "pose_id_measurement_impl.h"

#endif