#ifndef PERSON_HYPOTHESIS_IMPL_H
#define PERSON_HYPOTHESIS_IMPL_H

#include "mc3d_common.h"
#include "person_hypothesis.h"

namespace mc3d
{
    PersonHypothesis::PersonHypothesis(SplineWindow &spline, const PoseParameter &pose_parameter) : 
        spline(spline),
        pose_parameter(pose_parameter),
        spline_parameters(torch::zeros({static_cast<long>(pose_parameter.nb_keypoints),
            static_cast<long>(spline.get_nb_basis()),
            3},
            TensorRealTypeOption.requires_grad(true)
        )),
        scale_factor(torch::tensor({{1.0}}, TensorRealTypeOption.requires_grad(true)).reshape({1, 1})),
        scale_mvn(
            torch::tensor({{1.0}}, TensorRealTypeOption.requires_grad(false)).reshape({1, 1}),
            torch::eye(1, TensorRealTypeOption.requires_grad(false))
        )
    { }

    // sum all priors
    inline Tensor PersonHypothesis::parameter_log_prior() const
    {
        return smoothness_log_prior() + limb_length_log_prior() + scale_limb_length_log_prior();
    }

    // use  spline.evaluate with slice of spline_parameters and cam.transform
    Tensor PersonHypothesis::predict(size_t limb_id, RealType time, const Camera &camera) const
    {
        return camera.transform_3d_to_2d(
            spline.evaluate(
                time, spline_parameters.index({Slice(limb_id, limb_id + 1), Slice(), Slice()})
            ).transpose(0, 1)
        );
    }

    // use spline project_forward_spline_parameter on slices
    void PersonHypothesis::project_forward_spline_parameters()
    {
        spline.project_forward_spline_parameter(spline_parameters);
    }

    // use spline.spline_smoothness_log_prior
    Tensor PersonHypothesis::smoothness_log_prior() const
    {
        Tensor smoothness_log_prior_sum{torch::zeros({1}, TensorRealTypeOption.requires_grad(false))};

        for (int i = 0; i < pose_parameter.nb_keypoints; i++)
        {
            smoothness_log_prior_sum += spline.spline_smoothness_log_prior(spline_parameters.index({Slice(i, i + 1), Slice(), Slice()}));
        }

        return smoothness_log_prior_sum;
    }

    // use pose_parameter.average_limb_lengths
    Tensor PersonHypothesis::limb_length_log_prior() const
    {
        Tensor prob_sum{torch::zeros({1}, TensorRealTypeOption.requires_grad(false))};
        RealType time = 0.0;

        for (int i = 0; i < pose_parameter.edges.size(); i++)
        {
            auto &edge = pose_parameter.edges[i];
            MultivariateNormal mvn(
                torch::tensor({pose_parameter.average_limb_lengths[i]}, TensorRealTypeOption.requires_grad(false)).reshape({1, 1}),
                scale_factor);
            prob_sum += mvn.log_prob((
                spline.evaluate(time, spline_parameters.index({Slice(edge.first, edge.first + 1), Slice(), Slice()}))
                - spline.evaluate(time, spline_parameters.index({Slice(edge.second, edge.second + 1), Slice(), Slice()})))
                .norm(2).reshape({1, 1}));
        }

        return prob_sum;
    }

    Tensor PersonHypothesis::scale_limb_length_log_prior() const
    {
        return scale_mvn.log_prob(scale_factor);
    }
}

#include "person_hypothesis_impl.h"

#endif