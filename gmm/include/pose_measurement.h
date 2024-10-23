#ifndef POSE_MEASUREMENT_H
#define POSE_MEASUREMENT_H

#include "mc3d_common.h"
#include "camera.h"

namespace mc3d
{
    struct PoseMeasurement
    {
        RealType time;
        const Camera& camera;
        vector<size_t> limb_ids;
        Tensor values; //assert(values.sizes() == {limb_ids.size(),2})
    };
}

#endif