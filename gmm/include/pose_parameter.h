#ifndef POSE_PARAMETER_H
#define POSE_PARAMETER_H

#include "mc3d_common.h"

namespace mc3d
{
    struct PoseParameter
    {
        size_t nb_keypoints;
        vector<std::pair<size_t, size_t>> edges;
        vector<RealType> average_limb_lengths; // index according to edges
    };
}

#endif