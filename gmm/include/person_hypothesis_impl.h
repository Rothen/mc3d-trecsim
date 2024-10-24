#ifndef PERSON_HYPOTHESIS_IMPL_H
#define PERSON_HYPOTHESIS_IMPL_H

#include "mc3d_common.h"
#include "person_hypothesis.h"

namespace mc3d
{
    PersonHypothesis::PersonHypothesis(SplineWindow &spline, const PoseParameter &pose_parameter) : 
        spline(spline),
        pose_parameter(pose_parameter)
    {
        spline_parameter = torch::zeros({
            static_cast<long>(pose_parameter.nb_keypoints), 
            static_cast<long>(spline.nb_spline_parameter),
            3
        }, TensorRealTypeOption.requires_grad(true));
    }

    Tensor PersonHypothesis::parameter_log_prior() const
    {

    }

    Tensor PersonHypothesis::predict(size_t limb_id, RealType time, const Camera &camera) const
    {

    }

    void PersonHypothesis::project_forward_spline_parameter()
    {
        
    }

    Tensor PersonHypothesis::smoothness_log_prior() const
    {

    }

    Tensor PersonHypothesis::limb_length_log_prior() const
    {

    }

    Tensor PersonHypothesis::scale_limb_length_log_prior() const
    {

    }
}

#include "person_hypothesis_impl.h"

#endif