#ifndef PERSON_HYPOTHESIS_H
#define PERSON_HYPOTHESIS_H

#include "mc3d_common.h"
#include "spline_window.h"
#include "pose_parameter.h"
#include "camera.h"

namespace mc3d
{
    class PersonHypothesis
    {
    public:
        PersonHypothesis(SplineWindow& spline,const PoseParameter& pose_parameter);

    public:
        Tensor parameter_log_prior() const; //sum all priors
        Tensor predict(size_t limb_id, RealType time, const Camera& camera) const; //use  spline.evaluate with slice of spline_parameter and cam.transform
        void project_forward_spline_parameter(); // use spline project_forward_spline_parameter on slices
    protected:
        Tensor smoothness_log_prior() const; // use spline.spline_smoothness_log_prior
        Tensor limb_length_log_prior() const; //use pose_parameter.average_limb_lengths
        Tensor scale_limb_length_log_prior() const;

    protected:
        SplineWindow &spline;
        const PoseParameter& pose_parameter;
        Tensor spline_parameter; // spline_parameter.sizes() == {pose_parameter.nb_keypoints,nb_spline_parameter,3} requires_grad
    };
}

#include "person_hypothesis_impl.h"

#endif